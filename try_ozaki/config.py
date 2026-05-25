"""Configuration and constants."""

import os
from pathlib import Path

# Run:ai
RUNAI_PROJECT = os.environ.get("RUNAI_PROJECT", "osu-default")
RUNAI_IMAGE = os.environ.get("RUNAI_IMAGE", "nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04")

# S3 — accessed from app host via boto3 (profile dirkcli, us-east-1)
S3_BUCKET = os.environ.get("OZAKI_S3_BUCKET", "runai-peterdir")
S3_REGION = os.environ.get("OZAKI_S3_REGION", "us-east-1")
S3_PREFIX = os.environ.get("OZAKI_S3_PREFIX", "try-ozaki-jobs")

# Run:ai datasource name (pre-configured in the cluster, no credential injection needed)
RUNAI_DATASOURCE = os.environ.get("OZAKI_RUNAI_DATASOURCE", "runai-peterdir")
# Mount path inside the worker container
DATASOURCE_MOUNT = os.environ.get("OZAKI_DATASOURCE_MOUNT", "/mnt/runai-peterdir")

# GPU job
GPU_REQUEST = int(os.environ.get("OZAKI_GPU_REQUEST", "1"))
JOB_TIMEOUT_SECS = int(os.environ.get("OZAKI_JOB_TIMEOUT", "600"))
JOB_POLL_INTERVAL = int(os.environ.get("OZAKI_POLL_INTERVAL", "10"))

# Ozaki libraries
OZIMMU_REPO = "https://github.com/enp1s0/ozIMMU"
ACCELERATOR_REPO = "https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU"

# Results
RESULTS_BASE = Path(os.environ.get("OZAKI_RESULTS_BASE", "/tmp/try-ozaki-results"))
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
