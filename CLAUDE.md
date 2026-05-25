# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`try-ozaki` is a Python package that automatically detects FP64 hotspots in source code, rewrites them using the **Ozaki Scheme** (FP64 → FP32 emulation via INT8/FP32 Tensor Cores), submits both the original and transformed versions as GPU jobs, and compares numerical output and performance.

## Package & Distribution

- Python **3.12** minimum
- Published to **PyPI** as `try-ozaki`
- Run:ai CLI is a pip dependency; users must run `runai login` on the host before invoking `try-ozaki` (token management is out of scope for now)

## Architecture

The package has two entry points over a shared pipeline:

- **CLI** (`try-ozaki <local-path>`) — operates on a local directory
- **Web** (`uvicorn try_ozaki.app:app`) — FastAPI app; accepts a GitHub URL, clones to a temp session dir, then runs the same pipeline

Both converge at the same analysis → rewrite → submit → compare flow.

### Pipeline stages

1. **AST analysis** *(app host)* — language-specific traversal to find FP64 hotspots (triple-nested loops over `double`/`REAL(8)`/`f64` arrays, BLAS DGEMM calls). Language priority: Fortran → C → C++ → Rust → Julia → Go.
2. **Rewrite** *(app host)* — **Claude Code** performs the rewrite against the session's cloned/local source tree. AST hotspots are passed as context; Claude Code generates the Ozaki-I or Ozaki-II wrapper code in-place. This mirrors how `codecheck` invokes Claude Code against a cloned repo.
3. **Job submission** *(app host → cluster)* — default: Run:ai CLI (Python dependency). Optional: SLURM via SSH/SCP (upload script to `~/temp/` on submit node, run `sbatch`).
4. **Build & execution** *(GPU worker node)* — worker compiles and links against `ozIMMU` / `accelerator_for_ozIMMU`, then runs both FP64 and Ozaki-emulated binaries.
5. **Validation** *(app host)* — collect output tensors from both pipelines, compare: max absolute error, relative error, wall-clock time, speedup ratio. Pass/fail against a per-job user-defined error tolerance.

### Web app (FastAPI)

Modeled on `~/gh/codecheck`. Key patterns to follow from that repo:
- Session isolation: each job gets a temp dir, cleaned up after completion or timeout (2h default)
- Streaming output via SSE (`StreamingResponse`) for compilation logs, job status, and results
- In-memory session map: `session_id → {tmp_dir, repo_dir, created}`
- **Public**, no authentication required (deployment is behind the firewall, so no abuse controls needed)
- **Deployment host**: runs on **appmotel**
- **Job queueing**: no app-side queue — submit all jobs straight to Run:ai and let its scheduler handle backpressure
- **Results retention**: 30 days, served at `try-ozaki.xxxxx/results/<job-id>`. Cleanup handled by an **app-side cron job** (not S3 lifecycle policy) so the app can extend retention for specific jobs if needed.

### Result delivery

| Mode | Delivery |
|---|---|
| Web (GitHub URL) | Auto-open a pull request on the user's repo via the `gh` CLI (running locally on the app host). PRs come from a **single shared bot account** — all users see the same author. |
| CLI in a git repo | Create a new branch `ozaki-<short-id>` with the rewrite committed |
| CLI not in a git repo | Emit a `.patch` file the user can apply manually |

### Result transfer (worker → app host)

GPU worker writes output tensors, timing data, and logs via a **Run:ai S3 datasource** that is pre-configured in the cluster and mounted into every job container automatically. No AWS credentials are injected into worker containers.

**Run:ai S3 datasource: `runai-peterdir`**
- Datasource name: `runai-peterdir` (scope: Project `osu-default`)
- Backing bucket: `s3://runai-peterdir` (region: `us-east-1`)
- Mount path inside container: `/mnt/runai-peterdir`
- Attached to every job via: `--datasource type=s3,name=runai-peterdir`
- The app host uploads job scripts and source archives to `s3://runai-peterdir/try-ozaki-jobs/<job-id>/` using its own AWS credentials (profile `dirkcli`, `us-east-1`)
- The worker reads from `/mnt/runai-peterdir/try-ozaki-jobs/<job-id>/` and writes results back to the same mount path
- The app host then reads results from S3 directly to validate

### Cluster backends

**Run:ai (default)**
- Run:ai CLI is a pip dependency; installed with the package
- Host machine needs SSH access to the Run:ai login node
- Users must authenticate (`runai login`) on the host before running `try-ozaki`
- Run:ai project/namespace is **configurable per job submission**, default: `osu-default`
- Worker container image: `nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Jobs submitted programmatically via the CLI from within the app process
- S3 datasource `runai-peterdir` is attached to every job — no AWS key injection needed in containers

**SLURM (optional)**
- Disabled by default; enabled via config/env var
- SSH/SCP to a designated submit node, upload job script to `~/temp/`, execute via `sbatch`
- See `~/gh/slurm2runai` for SLURM→Run:ai script patterns

### Ozaki kernel libraries

These must be installed on the GPU **worker node** (not the app host) as part of job setup:
- [`ozIMMU`](https://github.com/enp1s0/ozIMMU) — base INT8 Tensor Core Ozaki GEMM
- [`accelerator_for_ozIMMU`](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU) — patches providing `_EF`, `_RN`, `_H` variants and n-blocking

## Key References

- arXiv:2504.08009 — Ozaki Scheme II (CRT/modular)
- arXiv:2508.00441 — DGEMM without FP64 using Ozaki + FP8 Tensor Cores ([HTML](https://arxiv.org/html/2508.00441v3), [PDF](https://arxiv.org/pdf/2508.00441v3))
- [CERN/NVIDIA Openlab slides (July 2025)](https://indico.cern.ch/event/1538409/contributions/6522024/attachments/3097817/5488258/OZAKI_slide_CERN.pdf) — GPU throughput benchmarks across RTX 4090 → GB200
- [NVIDIA cuBLAS blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) — cuBLAS internal FP emulation approach

## Related Repositories

- `~/gh/codecheck` — reference FastAPI architecture (session handling, SSE streaming, repo cloning)
- `~/gh/slurm2runai` — Run:ai CLI usage and SLURM job submission patterns
