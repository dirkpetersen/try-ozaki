"""Core pipeline: async generator that yields Events for all stages.

Both the CLI driver (cli.py) and the FastAPI app (app.py) consume this
generator — the CLI prints events as plain text, the web app forwards them
as SSE.  The logic lives here exactly once.

Stage overview:
  1. Analyze     — regex hotspot detection (fast, no LLM)
  2. Rewrite     — Claude Code CLI invoked in the ozaki work-dir, real-time stream
  3. Upload      — pack + upload src.tar.gz (+ ozimmu.tar.gz if CUDA < 13.2) to S3
  4. Submit      — runai training standard submit
  5. Monitor     — async poll loop with phase-change events, heartbeat, image-pull
                   detection, live S3 log tail, 30s pre-timeout warning
  6. Results     — download from S3, numerical validation, report
"""

import asyncio
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import boto3

from .analyzer import analyze, Hotspot
from .claude_runner import _build_env, stream_claude_cli
from .config import (
    RUNAI_PROJECT, RUNAI_IMAGE, S3_BUCKET, S3_REGION, S3_PREFIX,
    RUNAI_DATASOURCE, DATASOURCE_MOUNT, RESULTS_BASE,
    JOB_TIMEOUT_SECS, JOB_POLL_INTERVAL,
)
from .events import Event
from .rewriter import _write_wrapper_once   # still used to emit wrapper files
from .results import collect, report


# ── helpers ───────────────────────────────────────────────────────────────────

_TERMINAL = frozenset({
    "Succeeded", "Failed", "Completed", "Error", "Stopped",
    "succeeded", "failed", "completed", "error", "stopped",
})

_HEARTBEAT_INTERVAL = 60   # seconds between "still waiting" messages
_TIMEOUT_WARN_BEFORE = 30  # warn this many seconds before the deadline


def _upload_s3(local_path: Path, bucket: str, key: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(local_path), bucket, key)


def _pack_sources(original_dir: Path, ozaki_dir: Path, dest: Path) -> None:
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(original_dir, arcname="src/original")
        tar.add(ozaki_dir, arcname="src/ozaki")


def _pack_ozimmu(ozimmu_local: Path, dest: Path) -> bool:
    if not ozimmu_local.is_dir():
        return False
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(ozimmu_local, arcname="ozIMMU")
    return True


def _ensure_ozimmu_local(cache_parent: Path) -> Path:
    ozimmu_dir = cache_parent / "ozIMMU"
    if (ozimmu_dir / "CMakeLists.txt").exists():
        return ozimmu_dir
    cache_parent.mkdir(parents=True, exist_ok=True)
    env = _build_env()
    subprocess.run(
        ["gh", "repo", "clone", "enp1s0/ozIMMU", str(ozimmu_dir), "--", "--depth=1"],
        check=True, capture_output=True, env=env,
    )
    subprocess.run(
        ["gh", "repo", "clone", "enp1s0/cutf", str(ozimmu_dir / "src" / "cutf"), "--", "--depth=1"],
        check=True, capture_output=True, env=env,
    )
    return ozimmu_dir


def _get_job_details(job_name: str, project: str) -> dict:
    """Return dict with phase, node, pending_count (best-effort)."""
    env = _build_env()
    result = subprocess.run(
        ["runai", "workload", "describe", job_name, "-p", project],
        capture_output=True, text=True, check=False, env=env,
    )
    details: dict = {"phase": "Unknown", "node": "", "image_pulling": False}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Phase:"):
                details["phase"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Status:"):
                details["phase"] = stripped.split(":", 1)[1].strip()
            # Pick up node assignment from pod table line
            if "Running" in stripped and "dgx" in stripped.lower():
                parts = stripped.split()
                if len(parts) >= 2:
                    details["node"] = parts[1]
            # Detect image pull in events
            if "Pulling" in stripped and "image" in stripped.lower():
                details["image_pulling"] = True
            if "Pulled" in stripped and "image" in stripped.lower():
                details["image_pulling"] = False
    return details


def _s3_log_tail(bucket: str, key: str, region: str, last_line: int) -> tuple[list[str], int]:
    """Download job.log from S3 and return new lines since last_line."""
    try:
        s3 = boto3.client("s3", region_name=region)
        obj = s3.get_object(Bucket=bucket, Key=key)
        text = obj["Body"].read().decode("utf-8", errors="replace")
        lines = text.splitlines()
        new = lines[last_line:]
        return new, len(lines)
    except Exception:
        return [], last_line


def _submit_runai(job_name: str, inline_cmd: str, project: str, image: str,
                  datasource: str) -> None:
    cmd = [
        "runai", "training", "standard", "submit", job_name,
        "-p", project,
        "-i", image,
        "--image-pull-policy", "IfNotPresent",
        "--gpu-devices-request", "1",
        "--preemptibility", "preemptible",
        "--priority", "low",
        "--auto-deletion-time-after-completion", "24h",
        "--datasource", f"type=s3,name={datasource}",
        "--command", "--",
        "bash", "-c", inline_cmd,
    ]
    subprocess.run(cmd, check=True, env=_build_env())


# ── rewrite prompt ────────────────────────────────────────────────────────────

def _build_rewrite_prompt(hotspots: list[Hotspot], source_path: Path) -> str:
    lines = [
        "You are performing an automated Ozaki Scheme rewrite for the try-ozaki tool.",
        "",
        "TASK: Rewrite the FP64 DGEMM hotspots listed below so they call OZAKI_DGEMM",
        "instead of DGEMM. The wrapper module and backend shims have already been",
        "written into this directory as:",
        "  ozaki_wrapper.f90       — Fortran module defining OZAKI_DGEMM",
        "  ozaki_wrapper_cublas.cpp — cuBLAS FP64 emulation shim (CUDA >= 13.2)",
        "  ozaki_wrapper_ozimmu.cpp — ozIMMU fallback shim (CUDA < 13.2)",
        "",
        "RULES:",
        "  1. Replace every call to DGEMM (case-insensitive) with OZAKI_DGEMM.",
        "     Preserve all arguments exactly.",
        "  2. Add 'use ozaki_wrapper' to the program/subroutine that contains",
        "     the call, immediately before 'implicit none'. Add it only once.",
        "  3. For triple-nested DO loops that implement C(i,j) += A(i,k)*B(k,j),",
        "     comment out the loop body and replace with:",
        "       call OZAKI_DGEMM('N','N', N, N, N, 1.0d0, A, N, B, N, 0.0d0, C, N)",
        "     adjusting array names and dimension variable from the actual code.",
        "  4. Do NOT change anything outside the identified hotspot regions.",
        "  5. Do NOT modify CMakeLists.txt or any generated wrapper files.",
        "",
        "HOTSPOTS:",
    ]
    for h in hotspots:
        try:
            rel = h.file.relative_to(source_path)
        except ValueError:
            rel = h.file
        lines.append(f"  {rel}  line {h.start_line}  [{h.kind}]")
        if h.context:
            for ctx_line in h.context.splitlines()[:6]:
                lines.append(f"    | {ctx_line}")
    lines += [
        "",
        "When done, output a brief summary of what was changed.",
    ]
    return "\n".join(lines)


# ── main pipeline generator ───────────────────────────────────────────────────

async def run_pipeline(
    source_path: Path,
    *,
    project: str | None = None,
    image: str | None = None,
    s3_bucket: str | None = None,
    s3_region: str | None = None,
    s3_prefix: str | None = None,
    datasource: str | None = None,
    datasource_mount: str | None = None,
    tolerance: float = 1e-6,
    no_submit: bool = False,
    dry_run: bool = False,
    cmake_flags: str = "",
    session_id: str = "",
):
    """Async generator — yields Event objects for every observable action.

    CLI: asyncio.run(drive_cli(run_pipeline(...)))
    Web: StreamingResponse(sse_gen(run_pipeline(...)))
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:8]
    # Apply config defaults for any unset kwargs
    project        = project        or RUNAI_PROJECT
    image          = image          or RUNAI_IMAGE
    s3_bucket      = s3_bucket      or S3_BUCKET
    s3_region      = s3_region      or S3_REGION
    s3_prefix      = s3_prefix      or S3_PREFIX
    datasource     = datasource     or RUNAI_DATASOURCE
    datasource_mount = datasource_mount or DATASOURCE_MOUNT
    source_path = source_path.resolve()
    job_id = f"ozaki-{uuid.uuid4().hex[:8]}"

    yield Event("session_id", session_id)
    yield Event("status", f"Job ID: {job_id}")
    yield Event("status", f"Source: {source_path}")

    # ── Stage 1: Analyze ─────────────────────────────────────────────────────
    yield Event("stage", "Stage 1: Analyzing FP64 hotspots...")
    hotspots = analyze(source_path)
    if not hotspots:
        yield Event("status", "No FP64 hotspots found. Nothing to do.")
        yield Event("done", "")
        return

    yield Event("status", f"Found {len(hotspots)} hotspot(s):")
    for h in hotspots:
        try:
            rel = h.file.relative_to(source_path)
        except ValueError:
            rel = h.file
        yield Event("hotspot", f"{rel}:{h.start_line}  [{h.kind}]  ({h.language})")

    if dry_run:
        yield Event("status", "--dry-run: stopping after analysis.")
        yield Event("done", "")
        return

    # ── Stage 2: Prepare working dirs ────────────────────────────────────────
    work_dir = Path(tempfile.mkdtemp(prefix=f"try-ozaki-{job_id}-"))
    orig_dir = work_dir / "original"
    ozaki_dir = work_dir / "ozaki"
    shutil.copytree(source_path, orig_dir)
    shutil.copytree(source_path, ozaki_dir)

    # Remap hotspot paths to the ozaki copy
    ozaki_hotspots = [
        Hotspot(
            file=ozaki_dir / h.file.relative_to(source_path),
            kind=h.kind, language=h.language,
            start_line=h.start_line, end_line=h.end_line,
            context=h.context, vars=h.vars,
        )
        for h in hotspots
    ]

    # Write the wrapper files so Claude Code finds them when it inspects the dir
    for h in ozaki_hotspots:
        _write_wrapper_once(h.file.parent, h.language)

    # ── Stage 2: Rewrite via Claude Code ─────────────────────────────────────
    yield Event("stage", "Stage 2: Rewriting hotspots with Claude Code...")
    prompt = _build_rewrite_prompt(ozaki_hotspots, ozaki_dir)
    async for ev in stream_claude_cli(prompt, cwd=ozaki_dir):
        yield ev
        if ev.kind == "error":
            yield Event("status", "Rewrite step failed. Aborting.")
            shutil.rmtree(work_dir, ignore_errors=True)
            yield Event("done", "")
            return

    if no_submit:
        yield Event("status", f"--no-submit: rewritten sources at {ozaki_dir}")
        yield Event("done", "")
        return

    # ── Stage 3: Pack and upload to S3 ───────────────────────────────────────
    yield Event("stage", "Stage 3: Uploading sources to S3...")
    archive = work_dir / "src.tar.gz"
    _pack_sources(orig_dir, ozaki_dir, archive)
    job_s3_prefix = f"{s3_prefix}/{job_id}"

    yield Event("status", "Uploading src.tar.gz...")
    await asyncio.get_event_loop().run_in_executor(
        None, _upload_s3, archive, s3_bucket, f"{job_s3_prefix}/src.tar.gz", s3_region
    )
    yield Event("status", f"Uploaded s3://{s3_bucket}/{job_s3_prefix}/src.tar.gz")

    # Bundle ozIMMU (needed if worker CUDA < 13.2)
    ozimmu_cache = Path.home() / ".cache" / "try-ozaki"
    try:
        ozimmu_local = _ensure_ozimmu_local(ozimmu_cache)
        ozimmu_archive = work_dir / "ozimmu.tar.gz"
        if _pack_ozimmu(ozimmu_local, ozimmu_archive):
            yield Event("status", "Uploading ozimmu.tar.gz...")
            await asyncio.get_event_loop().run_in_executor(
                None, _upload_s3, ozimmu_archive, s3_bucket,
                f"{job_s3_prefix}/ozimmu.tar.gz", s3_region,
            )
            yield Event("status", "Uploaded ozimmu.tar.gz")
    except Exception as e:
        yield Event("status", f"Warning: could not bundle ozIMMU ({e}); worker will attempt git clone.")

    # Write and upload job script
    from .job_script import generate as gen_script
    script_content = gen_script(
        job_id=job_id,
        s3_prefix=s3_prefix,
        datasource_mount=datasource_mount,
        cmake_flags=cmake_flags,
    )
    script_path = work_dir / "job.sh"
    script_path.write_text(script_content)
    await asyncio.get_event_loop().run_in_executor(
        None, _upload_s3, script_path, s3_bucket,
        f"{job_s3_prefix}/job.sh", s3_region,
    )
    yield Event("status", "Uploaded job.sh")

    # ── Stage 4: Submit ───────────────────────────────────────────────────────
    yield Event("stage", "Stage 4: Submitting GPU job to Run:ai...")
    inline_cmd = f"bash {datasource_mount}/{s3_prefix}/{job_id}/job.sh"
    await asyncio.get_event_loop().run_in_executor(
        None, _submit_runai, job_id, inline_cmd, project, image, datasource,
    )
    yield Event("status", f"Submitted: {job_id}")
    yield Event("job_status", {"job_id": job_id, "phase": "Submitted", "node": ""})

    # ── Stage 5: Monitor ──────────────────────────────────────────────────────
    yield Event("stage", "Stage 5: Waiting for job completion...")

    deadline = time.time() + JOB_TIMEOUT_SECS
    warn_emitted = False
    last_phase = ""
    last_heartbeat = time.time()
    last_log_line = 0
    log_s3_key = f"{job_s3_prefix}/job.log"

    while True:
        now = time.time()

        # Pre-timeout warning
        remaining = deadline - now
        if remaining <= _TIMEOUT_WARN_BEFORE and not warn_emitted:
            warn_emitted = True
            yield Event("status", f"⚠ Job timeout in {int(remaining)}s — job may not complete.")

        if now >= deadline:
            yield Event("job_status", {
                "job_id": job_id,
                "phase": f"Timeout after {JOB_TIMEOUT_SECS}s",
                "node": "",
            })
            break

        # Poll Run:ai
        details = await asyncio.get_event_loop().run_in_executor(
            None, _get_job_details, job_id, project,
        )
        phase = details["phase"]

        # Emit event only when phase changes
        if phase != last_phase:
            msg = f"Job {job_id}: {phase}"
            if details["node"]:
                msg += f" on {details['node']}"
            if phase in ("Initializing",) and details.get("image_pulling"):
                msg += " (pulling container image…)"
            yield Event("job_status", {
                "job_id": job_id,
                "phase": phase,
                "node": details.get("node", ""),
            })
            yield Event("status", msg)
            last_phase = phase

        if phase in _TERMINAL:
            break

        # Heartbeat every 60s so connection stays alive and user sees progress
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
            elapsed = int(now - (deadline - JOB_TIMEOUT_SECS))
            yield Event("status", f"⏳ Still {phase}… (elapsed: {elapsed}s)")
            last_heartbeat = now

        # Tail job.log from S3 during Running
        if phase == "Running":
            new_lines, last_log_line = await asyncio.get_event_loop().run_in_executor(
                None, _s3_log_tail, s3_bucket, log_s3_key, s3_region, last_log_line,
            )
            for log_line in new_lines:
                yield Event("log_line", log_line)

        await asyncio.sleep(JOB_POLL_INTERVAL)

    # Final log flush
    new_lines, _ = await asyncio.get_event_loop().run_in_executor(
        None, _s3_log_tail, s3_bucket, log_s3_key, s3_region, last_log_line,
    )
    for log_line in new_lines:
        yield Event("log_line", log_line)

    # ── Stage 6: Results ─────────────────────────────────────────────────────
    yield Event("stage", "Stage 6: Collecting results from S3...")
    results_dir = RESULTS_BASE / job_id
    validation = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: collect(
            job_id=job_id,
            runai_status=last_phase,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            local_dir=results_dir,
            error_tolerance=tolerance,
            s3_region=s3_region,
        ),
    )

    yield Event("report", report(validation))
    yield Event("done", "pass" if validation.passed else "fail")

    shutil.rmtree(work_dir, ignore_errors=True)
