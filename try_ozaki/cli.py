"""try-ozaki CLI entry point.

Usage:
    try-ozaki <local-path> [options]

Drives the shared pipeline generator and prints events to stdout in real time.
"""

import argparse
import asyncio
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env files: project root first, then user override wins."""
    try:
        from dotenv import load_dotenv
        load_dotenv()                                                      # ./.env
        load_dotenv(Path.home() / ".try-ozaki" / ".env", override=True)   # user wins
    except ImportError:
        pass


def _load_config():
    from .config import (
        RUNAI_PROJECT, RUNAI_IMAGE, S3_BUCKET, S3_REGION, S3_PREFIX,
        RUNAI_DATASOURCE, DATASOURCE_MOUNT,
    )
    return dict(
        project=RUNAI_PROJECT, image=RUNAI_IMAGE,
        s3_bucket=S3_BUCKET, s3_region=S3_REGION, s3_prefix=S3_PREFIX,
        datasource=RUNAI_DATASOURCE, datasource_mount=DATASOURCE_MOUNT,
    )


async def _drive(args: argparse.Namespace) -> int:
    from .events import terminal_format
    from .pipeline import run_pipeline

    cfg = _load_config()

    exit_code = 0
    async for event in run_pipeline(
        source_path=args.path,
        project=args.project or cfg["project"],
        image=args.image or cfg["image"],
        s3_bucket=args.s3_bucket or cfg["s3_bucket"],
        s3_region=args.s3_region or cfg["s3_region"],
        s3_prefix=args.s3_prefix or cfg["s3_prefix"],
        datasource=args.datasource or cfg["datasource"],
        datasource_mount=args.datasource_mount or cfg["datasource_mount"],
        tolerance=args.tolerance,
        no_submit=args.no_submit,
        dry_run=args.dry_run,
        cmake_flags=args.cmake_flags,
    ):
        line = terminal_format(event)
        if line is not None:
            print(line, flush=True)
        if event.kind == "done" and event.data == "fail":
            exit_code = 1
        if event.kind == "error":
            exit_code = 1

    return exit_code


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="try-ozaki",
        description="Rewrite FP64 hotspots with the Ozaki Scheme and validate on GPU.",
    )
    parser.add_argument("path", type=Path, help="Local source directory to analyze")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Max relative error threshold (default: 1e-6)")
    parser.add_argument("--project", default="",
                        help="Run:ai project (overrides config default)")
    parser.add_argument("--image", default="",
                        help="Container image for GPU worker")
    parser.add_argument("--s3-bucket", default="",
                        help="S3 bucket for job artifacts")
    parser.add_argument("--s3-region", default="",
                        help="S3 region")
    parser.add_argument("--s3-prefix", default="",
                        help="S3 key prefix")
    parser.add_argument("--datasource", default="",
                        help="Run:ai datasource name")
    parser.add_argument("--datasource-mount", default="",
                        help="Mount path inside container")
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

    sys.exit(asyncio.run(_drive(args)))


if __name__ == "__main__":
    main()
