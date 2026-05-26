"""Fetch results from S3 and compute numerical validation metrics."""

import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    job_id: str
    status: str                # runai terminal status
    elapsed_orig_ms: int = 0
    elapsed_ozaki_ms: int = 0
    speedup: float = 0.0
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    out_orig: str = ""
    out_ozaki: str = ""
    passed: bool = False
    error_tolerance: float = 1e-6


def _s3_download(bucket: str, key: str, local: Path, region: str = "us-east-1") -> bool:
    import boto3
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.download_file(bucket, key, str(local))
        return True
    except Exception:
        return False


def _parse_floats(text: str) -> list[float]:
    """Extract floats from RESULT: lines only.

    Programs emit:  RESULT C11= <val>  CNN= <val>  sum= <val>
    Comparing only these lines avoids false errors from different timing numbers.
    Falls back to all floats in the output if no RESULT: lines are found.
    """
    result_lines = [l for l in text.splitlines() if l.startswith("RESULT")]
    source = "\n".join(result_lines) if result_lines else text
    return [float(m) for m in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", source)]


def collect(
    job_id: str,
    runai_status: str,
    s3_bucket: str,
    s3_prefix: str,
    local_dir: Path,
    error_tolerance: float = 1e-6,
    s3_region: str = "us-east-1",
) -> ValidationResult:
    local_dir.mkdir(parents=True, exist_ok=True)
    key_base = f"{s3_prefix}/{job_id}"

    result = ValidationResult(job_id=job_id, status=runai_status, error_tolerance=error_tolerance)

    # Download artifacts
    for fname in ("out_orig.txt", "out_ozaki.txt", "timing.json", "job.log"):
        dest = local_dir / fname
        ok = _s3_download(s3_bucket, f"{key_base}/{fname}", dest, region=s3_region)
        if not ok:
            print(f"[try-ozaki] Warning: could not download {fname} from S3", file=sys.stderr)

    # Parse timing
    timing_file = local_dir / "timing.json"
    if timing_file.exists():
        try:
            timing = json.loads(timing_file.read_text())
            result.elapsed_orig_ms = timing.get("elapsed_orig_ms", 0)
            result.elapsed_ozaki_ms = timing.get("elapsed_ozaki_ms", 0)
            if result.elapsed_orig_ms > 0 and result.elapsed_ozaki_ms > 0:
                result.speedup = result.elapsed_orig_ms / result.elapsed_ozaki_ms
        except Exception as e:
            print(f"[try-ozaki] Warning parsing timing: {e}", file=sys.stderr)

    # Read outputs
    out_orig_file = local_dir / "out_orig.txt"
    out_ozaki_file = local_dir / "out_ozaki.txt"
    if out_orig_file.exists():
        result.out_orig = out_orig_file.read_text(errors="replace")
    if out_ozaki_file.exists():
        result.out_ozaki = out_ozaki_file.read_text(errors="replace")

    # Numerical comparison
    if result.out_orig and result.out_ozaki:
        orig_nums = _parse_floats(result.out_orig)
        ozaki_nums = _parse_floats(result.out_ozaki)
        n = min(len(orig_nums), len(ozaki_nums))
        if n > 0:
            abs_errors = [abs(a - b) for a, b in zip(orig_nums[:n], ozaki_nums[:n])]
            rel_errors = [
                abs(a - b) / max(abs(a), 1e-300)
                for a, b in zip(orig_nums[:n], ozaki_nums[:n])
            ]
            result.max_abs_error = max(abs_errors)
            result.max_rel_error = max(rel_errors)
            result.passed = (result.max_rel_error <= error_tolerance)

    return result


def report(r: ValidationResult) -> str:
    lines = [
        f"╔══ try-ozaki results ═══════════════════════════════════════",
        f"║  Job ID      : {r.job_id}",
        f"║  Run:ai status: {r.status}",
        f"║  Original    : {r.elapsed_orig_ms:,} ms",
        f"║  Ozaki       : {r.elapsed_ozaki_ms:,} ms",
        f"║  Speedup     : {r.speedup:.2f}×" if r.speedup else "║  Speedup     : N/A",
        f"║  Max abs err : {r.max_abs_error:.3e}" if r.max_abs_error is not None else "║  Max abs err : N/A",
        f"║  Max rel err : {r.max_rel_error:.3e}" if r.max_rel_error is not None else "║  Max rel err : N/A",
        f"║  Tolerance   : {r.error_tolerance:.0e}",
        f"║  PASS/FAIL   : {'✓ PASS' if r.passed else '✗ FAIL'}",
        f"╚═══════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)
