"""try-ozaki CLI entry point.

Usage:
    try-ozaki <local-path> [--tolerance 1e-6] [--project osu-default]
              [--no-submit] [--dry-run]
"""

import argparse
import shutil
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

import boto3

from .analyzer import analyze, Hotspot
from .config import (
    RUNAI_PROJECT, RUNAI_IMAGE, S3_BUCKET, S3_REGION, S3_PREFIX,
    RUNAI_DATASOURCE, DATASOURCE_MOUNT, RESULTS_BASE,
)
from .job_script import generate as gen_script
from .rewriter import rewrite
from .results import collect, report
from .submitter import delete_job, stream_logs, submit_job, wait_for_job


def _pack_sources(original_dir: Path, ozaki_dir: Path, dest: Path) -> None:
    """Pack original/ and ozaki/ into a single src.tar.gz."""
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(original_dir, arcname="src/original")
        tar.add(ozaki_dir, arcname="src/ozaki")


def _upload_s3(local_path: Path, bucket: str, key: str, region: str = "us-east-1") -> None:
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(local_path), bucket, key)
    print(f"[try-ozaki] Uploaded s3://{bucket}/{key}", flush=True)


def run(
    source_path: Path,
    project: str = RUNAI_PROJECT,
    image: str = RUNAI_IMAGE,
    s3_bucket: str = S3_BUCKET,
    s3_region: str = S3_REGION,
    s3_prefix: str = S3_PREFIX,
    datasource: str = RUNAI_DATASOURCE,
    datasource_mount: str = DATASOURCE_MOUNT,
    tolerance: float = 1e-6,
    no_submit: bool = False,
    dry_run: bool = False,
    cmake_flags: str = "",
) -> int:
    """Main pipeline. Returns exit code (0=pass, 1=fail/error)."""

    source_path = source_path.resolve()
    job_id = f"ozaki-{uuid.uuid4().hex[:8]}"
    print(f"[try-ozaki] Job ID : {job_id}", flush=True)
    print(f"[try-ozaki] Source : {source_path}", flush=True)

    # ── 1. Analyze ────────────────────────────────────────────────────────────
    print("[try-ozaki] Stage 1: Analyzing FP64 hotspots...", flush=True)
    hotspots = analyze(source_path)
    if not hotspots:
        print("[try-ozaki] No FP64 hotspots found. Nothing to do.", flush=True)
        return 0

    print(f"[try-ozaki] Found {len(hotspots)} hotspot(s):", flush=True)
    for h in hotspots:
        rel = h.file.relative_to(source_path)
        print(f"  {rel}:{h.start_line}  [{h.kind}]  ({h.language})", flush=True)

    if dry_run:
        print("[try-ozaki] --dry-run: stopping after analysis.", flush=True)
        return 0

    # ── 2. Prepare working dirs ───────────────────────────────────────────────
    work_dir = Path(tempfile.mkdtemp(prefix=f"try-ozaki-{job_id}-"))
    orig_dir = work_dir / "original"
    ozaki_dir = work_dir / "ozaki"
    shutil.copytree(source_path, orig_dir)
    shutil.copytree(source_path, ozaki_dir)

    ozaki_hotspots = [
        Hotspot(
            file=ozaki_dir / h.file.relative_to(source_path),
            kind=h.kind, language=h.language,
            start_line=h.start_line, end_line=h.end_line,
            context=h.context, vars=h.vars,
        )
        for h in hotspots
    ]

    # ── 3. Rewrite ────────────────────────────────────────────────────────────
    print("[try-ozaki] Stage 2: Rewriting hotspots...", flush=True)
    modified = rewrite(ozaki_dir, ozaki_hotspots)
    for m in modified:
        print(f"  Modified: {m.relative_to(ozaki_dir)}", flush=True)

    if no_submit:
        print(f"[try-ozaki] --no-submit: rewritten sources at {ozaki_dir}", flush=True)
        return 0

    # ── 4. Pack & upload to S3 (app host uses its own AWS creds) ─────────────
    print("[try-ozaki] Stage 3: Uploading sources to S3...", flush=True)
    archive = work_dir / "src.tar.gz"
    _pack_sources(orig_dir, ozaki_dir, archive)
    job_s3_prefix = f"{s3_prefix}/{job_id}"
    _upload_s3(archive, s3_bucket, f"{job_s3_prefix}/src.tar.gz", region=s3_region)

    # ── 5. Generate & upload job script ──────────────────────────────────────
    script_content = gen_script(
        job_id=job_id,
        s3_prefix=s3_prefix,
        datasource_mount=datasource_mount,
        cmake_flags=cmake_flags,
    )
    script_path = work_dir / "job.sh"
    script_path.write_text(script_content)
    _upload_s3(script_path, s3_bucket, f"{job_s3_prefix}/job.sh", region=s3_region)

    # ── 6. Submit job (datasource provides S3 mount, no creds in container) ──
    print("[try-ozaki] Stage 4: Submitting GPU job...", flush=True)
    # Worker reads job.sh from the mount, no aws CLI needed
    inline_cmd = f"bash {datasource_mount}/{s3_prefix}/{job_id}/job.sh"

    submit_job(
        job_name=job_id,
        inline_command=inline_cmd,
        project=project,
        image=image,
        gpu=1,
        datasource=datasource,
    )

    # ── 7. Wait for completion ────────────────────────────────────────────────
    print("[try-ozaki] Stage 5: Waiting for job completion...", flush=True)
    final_status = wait_for_job(job_id, project=project)
    print(f"[try-ozaki] Final status: {final_status}", flush=True)

    # ── 8. Collect & validate results ────────────────────────────────────────
    print("[try-ozaki] Stage 6: Collecting results from S3...", flush=True)
    results_dir = RESULTS_BASE / job_id
    validation = collect(
        job_id=job_id,
        runai_status=final_status,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_region=s3_region,
        local_dir=results_dir,
        error_tolerance=tolerance,
    )

    print(report(validation), flush=True)

    log_file = results_dir / "job.log"
    if log_file.exists():
        print("\n[try-ozaki] ── Worker log (last 40 lines) ──────────────────", flush=True)
        for line in log_file.read_text(errors="replace").splitlines()[-40:]:
            print(line, flush=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    return 0 if validation.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="try-ozaki",
        description="Rewrite FP64 hotspots with the Ozaki Scheme and validate on GPU.",
    )
    parser.add_argument("path", type=Path, help="Local source directory to analyze")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Max relative error threshold (default: 1e-6)")
    parser.add_argument("--project", default=RUNAI_PROJECT,
                        help=f"Run:ai project (default: {RUNAI_PROJECT})")
    parser.add_argument("--image", default=RUNAI_IMAGE,
                        help="Container image for GPU worker")
    parser.add_argument("--s3-bucket", default=S3_BUCKET,
                        help=f"S3 bucket (default: {S3_BUCKET})")
    parser.add_argument("--s3-region", default=S3_REGION,
                        help=f"S3 region (default: {S3_REGION})")
    parser.add_argument("--s3-prefix", default=S3_PREFIX,
                        help=f"S3 key prefix (default: {S3_PREFIX})")
    parser.add_argument("--datasource", default=RUNAI_DATASOURCE,
                        help=f"Run:ai datasource name (default: {RUNAI_DATASOURCE})")
    parser.add_argument("--datasource-mount", default=DATASOURCE_MOUNT,
                        help=f"Mount path inside container (default: {DATASOURCE_MOUNT})")
    parser.add_argument("--cmake-flags", default="",
                        help="Extra CMake flags for worker builds")
    parser.add_argument("--no-submit", action="store_true",
                        help="Analyze and rewrite only, skip job submission")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, no rewrite or submission")

    args = parser.parse_args()

    if not args.path.is_dir():
        print(f"[try-ozaki] Error: '{args.path}' is not a directory.", file=sys.stderr)
        sys.exit(2)

    sys.exit(run(
        source_path=args.path,
        project=args.project,
        image=args.image,
        s3_bucket=args.s3_bucket,
        s3_region=args.s3_region,
        s3_prefix=args.s3_prefix,
        datasource=args.datasource,
        datasource_mount=args.datasource_mount,
        tolerance=args.tolerance,
        no_submit=args.no_submit,
        dry_run=args.dry_run,
        cmake_flags=args.cmake_flags,
    ))


if __name__ == "__main__":
    main()
