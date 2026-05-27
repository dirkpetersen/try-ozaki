"""Submit GPU jobs via Run:ai CLI and monitor to completion."""

import json
import subprocess
import sys
import time
from pathlib import Path

from .claude_runner import _build_env
from .config import RUNAI_PROJECT, RUNAI_IMAGE, GPU_REQUEST, JOB_TIMEOUT_SECS, JOB_POLL_INTERVAL, RUNAI_DATASOURCE


def _run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs: dict = {"check": check, "env": _build_env()}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def submit_job(
    job_name: str,
    inline_command: str,
    project: str = RUNAI_PROJECT,
    image: str = RUNAI_IMAGE,
    gpu: int = GPU_REQUEST,
    datasource: str = RUNAI_DATASOURCE,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Submit a training job and return the job name.

    inline_command: full shell command string to run inside the container.
    datasource: Run:ai datasource name to attach (provides S3 mount, no AWS creds needed).
    """
    cmd = [
        "runai", "training", "standard", "submit", job_name,
        "-p", project,
        "-i", image,
        "--image-pull-policy", "IfNotPresent",
        "--gpu-devices-request", str(gpu),
        "--preemptibility", "preemptible",
        "--priority", "low",
        "--auto-deletion-time-after-completion", "24h",
        "--datasource", f"type=s3,name={datasource}",
    ]
    # env vars must come BEFORE --command --
    if env_vars:
        for k, v in env_vars.items():
            cmd += ["-e", f"{k}={v}"]
    cmd += [
        "--command",
        "--",
        "bash", "-c", inline_command,
    ]

    print(f"[try-ozaki] Submitting job: {job_name}", flush=True)
    _run(cmd)
    return job_name


def get_job_status(job_name: str, project: str = RUNAI_PROJECT) -> str:
    """Return the current phase/status of a workload."""
    # Primary: table output — "Phase:  Running" is always present
    result = _run(
        ["runai", "workload", "describe", job_name, "-p", project],
        capture=True, check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.strip().startswith("Phase:"):
                return line.split(":", 1)[1].strip()
            if line.strip().startswith("Status:"):
                return line.split(":", 1)[1].strip()

    # Secondary: JSON output
    result_json = _run(
        ["runai", "workload", "describe", job_name, "-p", project, "--output", "json"],
        capture=True, check=False,
    )
    if result_json.returncode == 0 and result_json.stdout.strip():
        try:
            data = json.loads(result_json.stdout)
            if isinstance(data, list):
                data = data[0] if data else {}
            status = (
                data.get("status", {}).get("phase")
                or data.get("phase")
                or "Unknown"
            )
            if isinstance(status, dict):
                status = status.get("phase", "Unknown")
            return str(status)
        except Exception:
            pass

    return "Unknown"


def wait_for_job(
    job_name: str,
    project: str = RUNAI_PROJECT,
    timeout: int = JOB_TIMEOUT_SECS,
    poll: int = JOB_POLL_INTERVAL,
) -> str:
    """Poll until job reaches a terminal state. Returns final status string."""
    deadline = time.time() + timeout
    last_status = "Unknown"
    _TERMINAL = {"Succeeded", "Failed", "Completed", "Error", "Stopped",
                 "succeeded", "failed", "completed", "error", "stopped"}

    while time.time() < deadline:
        try:
            last_status = get_job_status(job_name, project)
        except Exception as e:
            print(f"[try-ozaki] Warning polling job: {e}", file=sys.stderr)

        print(f"[try-ozaki] Job {job_name} status: {last_status}", flush=True)

        if last_status in _TERMINAL:
            return last_status

        time.sleep(poll)

    return f"Timeout after {timeout}s (last: {last_status})"


def stream_logs(job_name: str, project: str = RUNAI_PROJECT) -> None:
    """Stream job logs to stdout (blocking)."""
    subprocess.run(
        ["runai", "workload", "logs", job_name, "-p", project, "--follow"],
        check=False, env=_build_env(),
    )


def delete_job(job_name: str, project: str = RUNAI_PROJECT) -> None:
    _run(
        ["runai", "workload", "delete", job_name, "-p", project],
        check=False, capture=True,
    )
