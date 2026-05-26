# try-ozaki

Automated tool that analyzes a source repository, rewrites intensive FP64 operations using the **Ozaki Scheme** (FP64 → FP32 emulation), and validates numerical parity and performance by running both the original and transformed code on a real GPU machine.

---

## Implementation Status

> Last end-to-end test: **2026-05-26**, job `ozaki-6d696c04`, node `dgx001` (NVIDIA H200), cluster `osu-default`.

### What worked

| Stage | Result |
|---|---|
| FP64 hotspot analysis (Fortran triple-nested loops, DGEMM calls) | ✅ Correctly detected 1 hotspot in `examples/ozaki-simple` |
| Rewriter — loop replaced with `OZAKI_DGEMM` + wrappers generated | ✅ Correct rewrite produced |
| Source + ozIMMU bundled and uploaded to S3 (`s3://runai-peterdir`) | ✅ Working — bundled on app host, no internet needed on worker |
| Run:ai job submission with datasource mount | ✅ Job scheduled and ran on dgx001 H200 |
| Job status polling | ✅ Working |
| **Original FP64 binary compiled and ran on H200** | ✅ **52.1 s total, ~1.66 GFLOP/s for 2048×2048 × 5 reps** |
| **ozIMMU built from source on worker (sm_90)** | ✅ Compiled for H200 architecture |
| **Ozaki binary linked and ran on H200** | ✅ **43.8 s total — 1.19× speedup over FP64** |
| Result files synced to S3, downloaded, validation report generated | ✅ All artifacts present |
| Test suite (12 tests: certik/matmul, ozaki-simple, synthetic FP64) | ✅ 12/12 pass |

### First real numbers (job `ozaki-6d696c04`, 2026-05-26)

```
Original FP64 (triple-nested loop):  52,057 ms   ~1.66 GFLOP/s
Ozaki (INT8 Tensor Core emulation):  43,830 ms   ~1.97 GFLOP/s (Rep 1: 49 GFLOP/s)
Speedup:  1.19×
```

Rep 1 of the Ozaki run hit **49.22 GFLOP/s** — a **30× burst** over baseline — because ozIMMU uses cuBLAS INT8 Tensor Cores internally. Reps 2–5 ran at normal speed because our current `beta` pre-scaling loop runs on CPU, not GPU, dominating the per-call overhead for repeated calls.

### What still needs fixing

#### 1. Numerical accuracy (FAIL)

Max relative error: **28.65** — completely wrong. Root cause: the `beta=0` pre-scaling in `ozaki_wrapper.cpp` zeros `C` on CPU, then ozIMMU accumulates into it with `beta=1`. But the Fortran test program initializes `C=0` itself before calling OZAKI_DGEMM, so the beta handling is correct in principle. The error is likely due to our Fortran rewrite passing wrong array layout or dimensions — Fortran is column-major, and the loop variable order `(j, i, k)` maps to `C = A * B` only in a specific orientation. The wrong `transa`/`transb` or swapped M/N would produce a numerically different result.

Fix: verify the rewriter emits `OZAKI_DGEMM('N','N', N, N, N, ...)` with correct array ordering for Fortran column-major layout, and add a small N=4 correctness test.

#### 2. Per-call overhead

The `beta` pre-scaling (`for` loop over C) runs on CPU for each of the 5 reps after the first. Replace with a `cudaMemset` or pass `beta` directly to ozIMMU once it supports it, or restructure the Fortran test to call DGEMM once with large N.

### What failed along the way (resolved)

| Issue | Root cause | Fix applied |
|---|---|---|
| `-lcudart` not found | CUDA lib not in linker path | `ldconfig /usr/local/cuda/lib64` after ozIMMU install |
| `-lozimmu` not found | Library name is `ozimmu` not `ozIMMU` | Fixed in CMakeLists |
| `cutf/cublas.hpp` not found | Worker has no internet; cutf is a submodule at `enp1s0/cutf` | Bundle ozIMMU+cutf as `ozimmu.tar.gz` on app host, upload to S3 |
| `cudaStream_t` not a type | ozimmu.hpp included before `cuda_runtime.h` | Added `#include <cuda_runtime.h>` first in shim |
| Segfault on Ozaki call | Fortran passes all args by reference; C shim treated scalars as values | Fixed shim to take `const int *` and dereference |

---

## Project Requirements

### Usage Modes

The tool ships as a **Python package** (`try-ozaki`) with two entry points:

| Mode | Input | Cluster target |
|---|---|---|
| **CLI** | Local directory path | Run:ai (default) or SLURM (optional) |
| **Web (FastAPI)** | GitHub repository URL | Run:ai (default) or SLURM (optional) |

The CLI operates on a local folder already on disk. The web interface clones a GitHub repo into a temporary session directory and then follows the same pipeline.

### Functional Requirements

1. **FP64 → FP32 conversion via Ozaki Scheme** — Detect FP64 hotspots in a target codebase and rewrite them using Ozaki-I (slice-based) or Ozaki-II (integer modular/CRT) emulation. Language support in priority order:
   - Fortran (primary)
   - C
   - C++
   - Rust
   - Julia
   - Go

2. **Ozaki library installation** — The package must install and link all required Ozaki kernel libraries on the GPU worker node as part of job setup, including:
   - [`ozIMMU`](https://github.com/enp1s0/ozIMMU) — INT8 Tensor Core-based Ozaki GEMM
   - Additional EFT/emulation libraries to be enumerated during implementation

3. **GPU validation** — Execute both the original (FP64) and transformed (Ozaki-emulated) code on a GPU machine and report:
   - Max absolute error and relative error between output tensors
   - Wall-clock execution time for both pipelines
   - Speedup ratio (FP64 baseline vs. Ozaki-emulated)
   - Per-job user-defined error tolerance threshold (pass/fail gate)

4. **Cluster job submission — Run:ai (default)**
   - The Run:ai CLI is a Python package dependency of `try-ozaki` and is installed alongside it
   - The FastAPI host (or CLI host) submits jobs programmatically via the Run:ai CLI without requiring a GPU locally
   - SSH access to the Run:ai login node is required from the host running `try-ozaki`

5. **Cluster job submission — SLURM (optional)**
   - Contact a designated SLURM submit node via SSH/SCP
   - Upload the generated job script to `~/temp/` on the submit node, then execute via `sbatch`
   - Configured via environment variable or config file; disabled by default

6. **Web application interface** — Python FastAPI app modeled on `codecheck`:
   - Accepts a GitHub repo URL as input
   - Streams transformation progress, compilation logs, and validation results to the browser in real time (SSE)
   - Session isolation: each job runs in a temporary directory, cleaned up after completion or timeout

### Non-Functional Requirements

- The web app and CLI host require no local GPU — all GPU work runs remotely via job submission
- Each conversion+validation run is fully isolated (temporary working directory per session)
- Streaming output for all long-running steps (AST analysis, compilation, job execution, result comparison)

---

## Related Repositories

Two sibling repositories are directly relevant to building and deploying this tool:

- **[codecheck](https://github.com/dirkpetersen/codecheck) (`~/gh/codecheck`)** — FastAPI web app that clones a GitHub repository and runs Claude Code against it with streaming output. The architecture of `try-ozaki`'s web interface follows this pattern.
- **[slurm2runai](https://github.com/dirkpetersen/slurm2runai) (`~/gh/slurm2runai`)** — Documents how to submit jobs to a Run:ai GPU cluster and convert SLURM batch scripts. Reference for the job submission layer in `try-ozaki`.

---

## Overview

Scientific researchers rely on double-precision (`FP64`) arithmetic to prevent catastrophic cancellation and maintain numerical stability. Modern GPU architectures, however, heavily favor low-precision execution units (`FP32`, `FP16`, `INT8`, `FP8` Tensor Cores) where throughput can be orders of magnitude higher than native `FP64`.

This tool bridges that gap — without requiring researchers to manually refactor their codebase — by automatically identifying compute-intensive `FP64` operations, rewriting them using **Ozaki Scheme Emulation**, and verifying numerical equivalence and performance on real GPU hardware.

---

## Theoretical Foundation: The Ozaki Scheme

The Ozaki Scheme is an **Error-Free Transformation (EFT)** framework for computing high-precision matrix multiplications using lower-precision compute kernels without accumulating rounding errors.

### Core Mechanism: Matrix Splitting & Reconstruction

Input matrices $A$ and $B$ are dynamically split into multiple lower-precision, bound-scaled "slices" such that cross-products of any two slices can be computed exactly within the significand bits of the target hardware (e.g., `FP32` or `INT8` Tensor Cores).

### Ozaki-I — Slice-Based Expansion

Elements are split based on controlled bitwidth boundaries. The high-precision result is reconstructed via weighted summation:

$$C = A \cdot B = \sum_{p=1}^{s_x} \sum_{q=1}^{s_y} A^{(p)} B^{(q)}$$

### Ozaki-II — Integer Modular / CRT-Based

A more advanced paradigm that maps floating-point matrices to scaled integer matrices, evaluates independent matrix products modulo a set of pairwise coprime moduli $(p_1, p_2, \dots, p_N)$, and recovers the exact result using the **Chinese Remainder Theorem (CRT)**. This significantly reduces the total number of required low-precision GEMM calls.

---

## Hotspot Identification

Converting an entire application indiscriminately creates unnecessary overhead. The tool isolates operations where computational density justifies the conversion.

### Target Identification Rules

1. **Algorithmic Footprint:** Target BLAS Level 3 operations — specifically `GEMM` — and high-density Level 2 operations (`GEMV`). The scheme scales optimally for $\mathcal{O}(n^3)$ operations on $\mathcal{O}(n^2)$ data.
2. **Loop Nesting Depth:** Isolate deeply nested loops (3 levels or deeper) where multi-dimensional arrays are accumulated into a target matrix.
3. **Variable Type Filtering:** Filter for variables declared as `REAL(8)`, `REAL(KIND=8)`, `DOUBLE PRECISION` (Fortran), `double` (C/C++), `f64` (Rust), `Float64` (Julia), `float64` (Go).

### AST Traversal Target Pattern (Fortran example)

```fortran
DO j = 1, N
    DO i = 1, M
        DO k = 1, K
            C(i,j) = C(i,j) + A(i,k) * B(k,j)
        END DO
    END DO
END DO
```

Or explicit library calls such as `CALL DGEMM(...)`.

---

## Automated Workflow Pipeline

```
[CLI: local path]  or  [Web: GitHub URL ➔ clone]
                              │
                    [AST Static Analysis]
                     Detect FP64 hotspots
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
[Pipeline A: FP64 Control]             [Pipeline B: Ozaki Emulation]
Keep original source                   Rewrite GEMM hotspots
Compile with standard flags            Inject Ozaki-I / Ozaki-II wrappers
                                       Install ozIMMU + deps on worker
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
               [Job Submission to GPU Cluster]
               Run:ai CLI (default) or SLURM via SSH
                              │
                              ▼
                   [Parallel Execution on GPU]
                              │
                              ▼
                   [Results & Validation]
            Max absolute error / relative error
            Wall-clock time, speedup ratio
            Pass/fail against per-job error threshold
```

---

## Reference Literature

- **Ozaki Scheme II Fundamentals:** *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique* (arXiv:2504.08009). Explains modular reduction formulations for high-precision emulation on NVIDIA GH200/RTX 4090.
- **Low-Precision Target Testing:** *DGEMM without FP64 Arithmetic — using FP64 Emulation and FP8 Tensor Cores with Ozaki Scheme* (arXiv:2508.00441) — [[HTML]](https://arxiv.org/html/2508.00441v3) [[PDF]](https://arxiv.org/pdf/2508.00441v3). Covers parameterization of slice counts to map `FP64` workloads onto hardware lacking substantial native `FP64` units.
- **Error Bound Optimization:** *Analysis of Floating-Point Matrix Multiplication Computed via Integer Arithmetic* (Netlib / Jack Dongarra et al., 2025). Provides Exponent-Span-Capacity (ESC) metrics to calculate the minimum slice count needed to guarantee target numerical precision.
- **cuBLAS Tensor Core FP Emulation:** [Unlocking Tensor Core Performance with Floating-Point Emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) (NVIDIA Developer Blog). Describes how cuBLAS itself uses FP emulation internally to exploit Tensor Cores for high-precision GEMM — directly relevant to the backend execution layer of the Ozaki-emulated pipeline.
- **CERN/NVIDIA Openlab Workshop Slides (July 2025):** [Emulating Matrix Multiplication Using Mixed-Precision Computation](https://indico.cern.ch/event/1538409/contributions/6522024/attachments/3097817/5488258/OZAKI_slide_CERN.pdf) — Ozaki, Uchino, Imamura. 25-slide deck covering both Ozaki Scheme I (slicing) and II (CRT/modular), with GPU throughput benchmarks across RTX 4090, H200, B200, and GB200. Includes INT8 Tensor Core emulation of FP64 with worked numerical examples.
- **ACM Paper — Ozaki Scheme on INT8 Matrix Units:** [Performance enhancement of the Ozaki Scheme on integer matrix multiplication units](https://dl.acm.org/doi/epdf/10.1145/3784828.3785017) (ACM, 2025). Covers the optimizations implemented in the `accelerator_for_ozIMMU` repository (error-free summation, alternative splitting, n-blocking). *(ACM login may be required.)*
- **accelerator_for_ozIMMU (RIKEN-RCCS):** [github.com/RIKEN-RCCS/accelerator_for_ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU) — CUDA/C++ patch library providing four performance enhancements over the base ozIMMU library: error-free summation (`ozIMMU_EF`), improved splitting (`ozIMMU_RN`), combined accuracy+performance variant (`ozIMMU_H`), and n-blocking for large matrices. A direct dependency candidate for the GPU worker node setup.
- **Emergent Mind — Ozaki Scheme II topic page:** [emergentmind.com/topics/ozaki-ii-scheme](https://www.emergentmind.com/topics/ozaki-ii-scheme) — Aggregates related papers including *Guaranteed DGEMM Accuracy via Extensions of the Ozaki Scheme* and *Emulation of Complex Matrix Multiplication based on CRT*. Useful for tracking new publications in this space.
