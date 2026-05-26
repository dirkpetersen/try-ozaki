# Discussion: The Ozaki Scheme, Declining GPU FP64, and What `try-ozaki` Should Be

This document captures the strategic rationale for `try-ozaki` and reassesses the project in light of NVIDIA's [CUDA Toolkit 13.0 Update 2](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) (October 2025), which integrates the Ozaki Scheme directly into cuBLAS.

---

## 1. The original concern: FP64 throughput is being de-prioritized in modern GPUs

Researchers in scientific computing have built decades of code on the assumption that double-precision arithmetic is a first-class GPU capability. That assumption is becoming hardware-dependent:

| GPU                     | Year | FP64 peak    | INT8 / FP4 peak          | FP64 fraction |
|-------------------------|------|--------------|--------------------------|---------------|
| V100                    | 2017 | 7.8 TFLOP/s  | 125 TFLOP/s (FP16 TC)    | ~6%           |
| A100                    | 2020 | 9.7 TFLOP/s  | 312 TFLOP/s (INT8 TC)    | ~3%           |
| H100                    | 2022 | 67 TFLOP/s   | 1,979 TFLOP/s (INT8 TC)  | ~3%           |
| H200                    | 2024 | 67 TFLOP/s   | 1,979 TFLOP/s (INT8 TC)  | ~3%           |
| **RTX 4090** (consumer) | 2022 | 1.3 TFLOP/s  | 1,457 TFLOP/s (INT8 TC)  | **~0.09%**    |
| B200 / GB200            | 2025 | ~80 TFLOP/s  | 36,000 TFLOP/s (FP4 TC)  | ~0.2%         |

Two patterns are clear:

1. The **HPC line** (V100 → A100 → H100 → H200) has preserved FP64, but it is a smaller and smaller fraction of total throughput each generation.
2. The **consumer / ML-focused line** (RTX, eventually likely Rubin) has effectively no FP64 — orders of magnitude less than INT8/FP4.

If NVIDIA's Rubin generation continues the Blackwell direction, scientific codes that depend heavily on FP64 will see their effective performance per dollar decline sharply, even as raw transistor counts continue to grow. The Ozaki Scheme is the leading technique for routing FP64 workloads onto INT8/FP8 Tensor Cores while preserving exact results.

---

## 2. What the Ozaki Scheme actually guarantees

The Ozaki Scheme is **not an approximation**. It is an Error-Free Transformation (EFT):

- Input matrices are dynamically split into low-precision slices such that each slice's significand fits within the target hardware's exact-multiply range.
- Cross-products of slices are computed exactly on INT8/FP32 Tensor Cores.
- The high-precision result is reconstructed via weighted summation, also exactly.
- For sufficiently many slices, the final result is **bit-identical to a native FP64 multiply**.

Our own end-to-end experiment on a single H200 (`examples/ozaki-simple`, 2048×2048 DGEMM, 5 reps) confirmed this: `max relative error = 0.000` against the cuBLAS FP64 reference, across all reps. The math works in practice exactly as the theory predicts.

The trade-off is the **slice count**. Each additional slice is one more INT8 GEMM call. ozIMMU's `auto` mode uses the input matrix exponent range to select the minimum slice count needed for IEEE 754 FP64 equivalent accuracy. For typical scientific matrices this is 3–8 slices.

---

## 3. The H200 result and what it means

In our test on dgx001:

```
cuBLAS FP64 (native):  623 ms    1,387 GFLOP/s
ozIMMU INT8 emulation: 976 ms      399 GFLOP/s
Speedup:               0.64×  (Ozaki is slower)
Max relative error:    0.000e+00
```

This is not a defeat — it is exactly the predicted outcome. The H200 has dedicated FP64 Tensor Cores and cuBLAS FP64 fully exploits them. On hardware that's optimized for FP64, native FP64 wins. The Ozaki advantage appears on hardware that **isn't** optimized for FP64.

Published benchmarks confirm this:

- **RTX 4090** (consumer, weak FP64): Ozaki on INT8 TCs delivers **30–50× speedup** vs native FP64 with zero numerical error.
- **GB200 NVL72** (Blackwell): NVIDIA reports up to **2.3× speedup** for FP64 GEMM via Ozaki emulation versus native FP64.
- **RTX PRO 6000 Blackwell** (workstation): Up to **13× speedup** for FP64 GEMM.

The trajectory of the FP64/INT8 ratio strongly suggests these speedup factors will *grow*, not shrink, on future generations.

---

## 4. The CUDA 13.0 Update 2 turning point (October 2025)

The NVIDIA blog post and the [official cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation) change the picture significantly. As of CUDA 13.0 Update 2, **cuBLAS itself implements the Ozaki Scheme natively** for FP64 matmul.

### Hardware support (correction — broader than the blog post implied)

The cuBLAS documentation specifies the actual compute-capability matrix:

| Emulation type      | Supported CCs                       | Includes (notable)                         | CUDA version |
|---------------------|-------------------------------------|--------------------------------------------|--------------|
| BF16x9 (FP32 emul.) | 10.0, 10.3                          | Blackwell only                             | 12.9+        |
| Fixed-Point (FP64)  | **8.x, 9.0, 10.0, 11.0, 12.x**      | **A100, H100, H200**, Blackwell, future    | 13.0u2+      |

This is the surprise: **FP64 emulation works on H100 and H200** (CC 9.0) and even **A100** (CC 8.0). The blog post features Blackwell because that's where the new hardware lives, but the implementation reaches back at least to Ampere. Our own H200 cluster qualifies — we just need to use the cuBLAS API instead of (or in addition to) ozIMMU directly.

### The cuBLAS emulation API

```c
// Math-mode flag for typed GEMM functions
cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH);

// Or set the compute type on cublasGemmEx() calls directly:
//   computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT

// Mantissa control (the key tuning knob)
cublasSetFixedPointEmulationMantissaControl(handle, CUBLAS_DYNAMIC_MANTISSA);  // ADP (default)
cublasSetFixedPointEmulationMaxMantissaBitCount(handle, 55);                   // tighter ceiling
cublasSetFixedPointEmulationMantissaBitOffset(handle, ...);                    // performance tuning

// Performance vs. eager strategy
cublasSetEmulationStrategy(handle, /* performant | eager */);

// Special values (Inf/NaN) handling
cublasSetEmulationSpecialValuesSupport(handle, ...);
```

Or with no code changes at all, via environment variables:

- `CUBLAS_EMULATE_DOUBLE_PRECISION=1` — turn on FP64 emulation globally
- `CUBLAS_EMULATION_STRATEGY=performant` (or `eager`)
- `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=55`

This means a user with an existing CUDA application that calls `cublasDgemm` can pick up Ozaki emulation by **setting an environment variable** before the binary launches, with no recompile. This is the lowest-effort path to the speedup.

### Mantissa control modes

Two control modes are exposed:

- **Dynamic mantissa control (default)** — the ADP framework analyzes inputs and picks the bit count needed for accuracy ≥ native FP64. Default ceiling: **79 mantissa bits**. Falls back to native FP64 if the matrix exponent range demands more bits than the ceiling allows.
- **Fixed mantissa control** — user pins a specific bit count (default 55). No fallback; pure performance/accuracy trade.

### Supported routines

FP emulation activates for:

- `cublasGemmEx()`, `cublasGemmBatchedEx()`, `cublasGemmStridedBatchedEx()`
- Typed-API GEMM (e.g. `cublasDgemm`) when `cublasSetMathMode()` is configured

Reported real-application results from NVIDIA:

| Application      | Hardware                | Speedup                          |
|------------------|-------------------------|----------------------------------|
| ecTrans (weather)| GB200 NVL72             | 2.4× SGEMM (BF16x9)              |
| BerkeleyGW       | B200                    | Significant ZGEMM speedup        |
| Quantum ESPRESSO | RTX PRO 6000 Blackwell  | 1.5× (ADP), ~3× (39 mantissa)    |

Heat maps in the blog show 2.3× DGEMM speedup on GB200 and **up to 13× on RTX PRO 6000 Blackwell**, across a wide range of matrix shapes, with no penalty on small matrices (cuBLAS heuristics fall back to native FP64).

### Known limitations of cuBLAS FP emulation

From the official documentation:

- **Bit-wise reproducibility is not guaranteed** with fixed-point emulation due to dynamic workspace allocation.
- **Large workspace requirements** — provide a pre-allocated workspace via `cublasSetWorkspace()` for best performance.
- **Not supported during CUDA Graph stream capture** without a pre-allocated workspace.

**This is the Ozaki Scheme arriving as a vendor-supported, drop-in capability for any code that already calls cuBLAS DGEMM/ZGEMM on Ampere, Hopper, or Blackwell.**

---

## 5. Honest reassessment: is custom Ozaki work still needed?

**For codes that already call cuBLAS DGEMM or ZGEMM on supported Blackwell hardware: no.** Just upgrade to CUDA 13.0 Update 2 and the speedup is automatic. There is no reason to maintain a parallel ozIMMU integration for those cases.

But several large categories of scientific code do **not** fall into that bucket:

1. **Legacy Fortran / C with hand-written triple-nested DGEMM-equivalent loops.** These never reach cuBLAS at all. They run on the CPU. The most expensive linear algebra in many older HPC codes still looks like:
   ```fortran
   do j = 1, N
     do i = 1, N
       do k = 1, N
         C(i,j) = C(i,j) + A(i,k) * B(k,j)
       end do
     end do
   end do
   ```
   No CUDA library can help this code until something rewrites it as a `DGEMM` call.

2. **Code calling CPU BLAS implementations** (Netlib reference, OpenBLAS, MKL CPU, ATLAS). These never touch a GPU. cuBLAS FP emulation is irrelevant until the call site is redirected.

3. **Code calling cuBLAS on hardware older than Ampere (V100, T4, P100, etc.).** The cuBLAS FP64 emulation supports compute capabilities 8.x and above (Ampere, Hopper, Blackwell, future), so A100 / H100 / H200 users *can* benefit just by upgrading CUDA. But Volta-class HPC clusters (V100) and older are excluded — they need the manual ozIMMU path or a hardware refresh. Pre-CUDA-13.0-U2 deployments also need the manual path until they upgrade.

4. **Code using GEMM implementations other than cuBLAS:** MAGMA, custom CUDA kernels, hipBLAS on AMD, oneAPI on Intel. None of these inherit the cuBLAS ADP framework.

5. **Code where researchers need finer accuracy/performance tuning** than cuBLAS exposes (e.g. application-specific knowledge that 39 mantissa bits is sufficient, like the Quantum ESPRESSO 3× speedup case). The cuBLAS ADP defaults to "as accurate as FP64"; getting more performance requires explicit precision configuration that varies per application.

6. **Codes that need to be evaluated** before users commit to a port. A researcher with 100K lines of Fortran needs to know whether the GEMM fraction of runtime is large enough to justify modernization, before doing the work.

---

## 6. What `try-ozaki` should be, given this landscape

The original framing — "rewrite hotspots to call ozIMMU directly" — is largely obsolete for codes that use cuBLAS on Blackwell. But the **diagnostic and modernization** value is large and growing:

### Core value proposition (revised)

`try-ozaki` should help researchers answer two concrete questions about their existing scientific codebase:

**Q1: How much of my runtime is FP64 GEMM that *could* be accelerated, if it were routed through cuBLAS?**

This requires:
- Source analysis to detect both DGEMM call sites *and* equivalent triple-nested loops
- A way to estimate the fraction of total runtime spent in those hotspots
- A clear report: "this code is 73% DGEMM-by-runtime — modernization will pay off"

**Q2: If I migrate this code to cuBLAS DGEMM (and thereby pick up cuBLAS FP emulation automatically on Blackwell), will my numerical results still be correct?**

This requires:
- Automated rewrite from CPU loops / OpenBLAS calls into `cublasDgemm`
- Side-by-side execution: original vs rewritten, both on GPU, compare outputs
- Pass/fail validation against a user-defined error tolerance — exactly what the current pipeline already does

What we built so far validates **the second half** of this end-to-end on a real GPU cluster: the analyze → rewrite → submit → validate loop works, and the numerical comparison was bit-exact (`max_err = 0.000`) once both pipelines ran on the GPU.

### Concrete pivot recommendations

1. **Default rewrite target should be `cublasDgemm`, not `ozimmu_dgemm`.** This is what users actually want most of the time. `ozIMMU` becomes an optional backend for non-Blackwell hardware, not the default.

2. **Add a profiling stage** that estimates how much of total runtime the detected hotspots represent. Without this, researchers can't justify the migration effort. A simple instrumented run that times the original DGEMM/loops gives this data.

3. **Add a hardware-aware recommendation** in the validation report. After running the test, output something like:
   - "On your H200: native cuBLAS FP64 was fastest (1,387 GFLOP/s)."
   - "On a B200 / GB200, cuBLAS would automatically use Ozaki FP64 emulation for an estimated 2.3× speedup with the same code."
   - "On RTX-class hardware, an explicit ozIMMU integration would deliver an estimated 13×+ speedup over native FP64."

4. **Keep the ozIMMU path as a fallback** for users on:
   - Pre-Blackwell GPUs (H100/H200/A100), where cuBLAS FP64 emulation isn't available
   - Older CUDA versions (< 13.0U2)
   - Codes that need precision-tuning beyond what cuBLAS ADP exposes

5. **Add CUDA version + compute-capability detection** to the worker job script. The decision matrix is now:
   - CUDA ≥ 13.0u2 **and** GPU CC ≥ 8.0 (Ampere or newer): just enable cuBLAS FP emulation via `cublasSetMathMode(CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)` or `CUBLAS_EMULATE_DOUBLE_PRECISION=1`. No source rewrite beyond routing through cuBLAS.
   - CUDA < 13.0u2 **or** GPU CC < 8.0 (Volta, Pascal): fall through to the explicit ozIMMU integration path.
   - Hand-written CPU loops or non-cuBLAS GEMM: rewrite to `cublasDgemm` regardless of hardware.

---

## 7. Recommendation for researchers asking about Vera Rubin

The concern about declining FP64 in future GPUs is **valid and well-timed**. The technical answer is:

1. **The Ozaki Scheme is real, it works, and it is now vendor-supported.** Codes ported to it get bit-exact FP64 results from INT8/FP8 Tensor Cores. This is no longer an experimental academic paper — it's in cuBLAS.

2. **For codes that already use cuBLAS DGEMM/ZGEMM, the migration to Vera Rubin is a CUDA upgrade, nothing more.** The ADP framework will automatically choose between native FP64 and emulated FP64 on whatever hardware is present.

3. **The codes that will struggle are those still using CPU loops, CPU BLAS, or non-cuBLAS GEMM implementations.** These need source-level modernization *before* the GPU upgrade matters. That is the gap `try-ozaki` should fill.

4. **The window to do this work calmly is now**, while H100/H200 hardware is still abundant and FP64 isn't yet a bottleneck. Researchers who wait until Rubin lands and FP64 is anemic will be doing the same migration under pressure.

The viability of the approach is no longer in question — NVIDIA shipped it. The remaining work is one of **code migration**: getting researchers' actual scientific codebases into the form where they can benefit from it. That is an automation and tooling problem, and it's the problem `try-ozaki` is well-positioned to solve.

---

## 8. Summary

| Question                                           | Answer                                                                          |
|----------------------------------------------------|---------------------------------------------------------------------------------|
| Is the Ozaki Scheme numerically sound?             | Yes — Error-Free Transformation. Bit-exact. Verified on H200.                   |
| Will FP64 keep declining on future GPUs?           | Likely yes on consumer / ML-class; HPC line uncertain (Rubin TBD).              |
| Does cuBLAS now do Ozaki natively?                 | Yes, in CUDA 13.0 Update 2 (Oct 2025).                                          |
| On which GPUs does cuBLAS FP64 emulation work?     | Ampere (CC 8.x), Hopper (9.0), Blackwell (10.x, 11.0, 12.x). Includes A100/H100/H200. |
| Can it be enabled without code changes?            | Yes — set `CUBLAS_EMULATE_DOUBLE_PRECISION=1` env var.                          |
| Do users still need to write Ozaki integration?    | Only for non-cuBLAS code, pre-Ampere GPUs, or precision-tuned cases.            |
| Does `try-ozaki` still have value?                 | Yes — for source-level modernization, profiling, and validation.                |
| What should `try-ozaki` default to rewriting to?   | `cublasDgemm`. ozIMMU is now a fallback for pre-Ampere hardware only.           |

The strategic position to communicate to researchers: **the Ozaki Scheme is no longer the question. The question is whether your code is in a form that can use it.** That is what this tool helps determine.

---

## 9. References

### NVIDIA primary sources

- **[cuBLAS Floating-Point Emulation — official documentation](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation)** — authoritative API reference for `cublasSetMathMode`, `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`, mantissa-control APIs, environment variables, supported compute capabilities, and known limitations.
- **[Unlocking Tensor Core Performance with Floating-Point Emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)** — NVIDIA developer blog post (October 2025) introducing FP64 emulation in CUDA 13.0 Update 2, with ecTrans / BerkeleyGW / Quantum ESPRESSO benchmark results and heat maps.
- **[CUDA Toolkit 13.0 Update 2 download](https://developer.nvidia.com/cuda-downloads)** — required version for FP64 emulation features.
- **[CUDA Library Samples — Emulation examples](https://github.com/NVIDIA/CUDALibrarySamples)** — runnable example code from NVIDIA showing how to invoke emulated GEMM.
- **[CUDA GPU Compute Capability matrix](https://developer.nvidia.com/cuda-gpus)** — map specific GPU SKUs to their compute capabilities (to determine cuBLAS emulation eligibility).

### Academic / theoretical foundations

- **arXiv:2504.08009** — *[Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique](https://arxiv.org/abs/2504.08009)*. The CRT/modular variant; reduces total low-precision GEMM calls relative to Ozaki-I.
- **arXiv:2508.00441** — *[DGEMM without FP64 Arithmetic — using FP64 Emulation and FP8 Tensor Cores with Ozaki Scheme](https://arxiv.org/abs/2508.00441)* ([HTML](https://arxiv.org/html/2508.00441v3) | [PDF](https://arxiv.org/pdf/2508.00441v3)). Parameterizes slice counts to map FP64 onto hardware lacking FP64 units. Reports 30–50× speedup on RTX 4090.
- **[Analysis of Floating-Point Matrix Multiplication Computed via Integer Arithmetic](https://www.netlib.org/lapack/lawnspdf/lawn324.pdf)** (Netlib LAWN, Dongarra et al., 2025) — Exponent-Span-Capacity (ESC) metric for selecting the minimum slice count to guarantee target precision.
- **[Performance enhancement of the Ozaki Scheme on integer matrix multiplication units](https://dl.acm.org/doi/epdf/10.1145/3784828.3785017)** (ACM, 2025) — describes the optimizations in `accelerator_for_ozIMMU`: error-free summation (`ozIMMU_EF`), improved splitting (`ozIMMU_RN`), combined variant (`ozIMMU_H`), and n-blocking.

### Reference implementations

- **[ozIMMU](https://github.com/enp1s0/ozIMMU)** (Mukunoki / enp1s0) — original CUDA/C++ INT8 Tensor Core Ozaki GEMM library. The reference implementation that `try-ozaki` currently links against on the GPU worker.
- **[accelerator_for_ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU)** (RIKEN-RCCS) — patch library adding `_EF`, `_RN`, `_H` performance variants and n-blocking for large matrices.
- **[cutf](https://github.com/enp1s0/cutf)** — CUDA Utility Functions, a header-only dependency of ozIMMU (note: lives at `enp1s0/cutf`, not `wmmae/cutf`).

### Workshop & talk material

- **[CERN/NVIDIA Openlab Workshop slides — Emulating Matrix Multiplication Using Mixed-Precision Computation (July 2025)](https://indico.cern.ch/event/1538409/contributions/6522024/attachments/3097817/5488258/OZAKI_slide_CERN.pdf)** — Ozaki, Uchino, Imamura. 25-slide deck covering Ozaki-I and Ozaki-II with GPU throughput benchmarks across RTX 4090, H200, B200, GB200.
- **[Energy-Efficient Supercomputing Through Tensor Core-Accelerated Mixed-Precision Computing and Floating-Point Emulation](https://www.nvidia.com/en-us/on-demand/)** — NVIDIA on-demand video on the energy/perf case for FP emulation.
- **[Precision Redefined: Unlocking and Delivering the Full Power of Modern GPUs for Scientific Computing](https://www.nvidia.com/en-us/on-demand/)** — NVIDIA conference talk slides on the broader FP emulation strategy.

### Background reading

- **[Emergent Mind — Ozaki Scheme II topic page](https://www.emergentmind.com/topics/ozaki-ii-scheme)** — aggregator of related papers, including *Guaranteed DGEMM Accuracy via Extensions of the Ozaki Scheme* and *Emulation of Complex Matrix Multiplication based on CRT*.
- **[NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)** — H100/H200 Tensor Core specifications, FP64 throughput numbers used in Section 1.
- **[NVIDIA Blackwell Architecture Whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)** — B200/GB200 specifications, FP4 / INT8 throughput.

### Project-internal documents

- [`README.md`](./README.md) — current implementation status, run results, what worked / what failed.
- [`CLAUDE.md`](./CLAUDE.md) — architecture, pipeline stages, S3 datasource setup, Run:ai cluster details.
- [`examples/ozaki-simple/`](./examples/ozaki-simple/) — minimal Fortran DGEMM benchmark used for end-to-end validation; covers both the cuBLAS native baseline and the ozIMMU rewrite.
