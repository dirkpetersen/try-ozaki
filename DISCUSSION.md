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

The NVIDIA blog post linked above changes the picture significantly. As of CUDA 13.0 Update 2, **cuBLAS itself implements the Ozaki Scheme natively** for FP64 matmul on supported hardware (GB200 NVL72, RTX PRO 6000 Blackwell, with more to follow):

- cuBLAS exposes an **Automatic Dynamic Precision (ADP) framework** that analyzes input matrices, picks the optimal slice count, and decides between native FP64 and emulated FP64 — automatically, with no source code changes.
- The default ADP guarantees accuracy **equal to or better than native FP64**.
- Users can optionally tune mantissa bits down (e.g. 55 / 47 / 39) to trade accuracy for further speedup.

Reported real-application results from NVIDIA:

| Application      | Hardware                | Speedup                          |
|------------------|-------------------------|----------------------------------|
| ecTrans (weather)| GB200 NVL72             | 2.4× SGEMM (BF16x9)              |
| BerkeleyGW       | B200                    | Significant ZGEMM speedup        |
| Quantum ESPRESSO | RTX PRO 6000 Blackwell  | 1.5× (ADP), ~3× (39 mantissa)    |

Heat maps in the blog show 2.3× to 13× DGEMM speedups across a wide range of matrix shapes, with no penalty on small matrices (cuBLAS heuristics fall back to native FP64).

**This is the Ozaki Scheme arriving as a vendor-supported, drop-in capability for any code that already calls cuBLAS DGEMM/ZGEMM.**

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

3. **Code calling cuBLAS on non-Blackwell hardware.** As of CUDA 13.0 Update 2, FP64 emulation is supported on GB200 NVL72 and RTX PRO 6000 Blackwell. Users on H100, H200, A100, V100, and consumer Ampere/Ada cards get nothing automatically. Older clusters need the manual integration path.

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

5. **Add CUDA version detection** to the worker job script. If `cuBLAS >= 13.0U2` and hardware is Blackwell, recommend just upgrading CUDA. Otherwise, fall through to the explicit Ozaki path.

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

| Question                                           | Answer                                                                  |
|----------------------------------------------------|-------------------------------------------------------------------------|
| Is the Ozaki Scheme numerically sound?             | Yes — Error-Free Transformation. Bit-exact. Verified on H200.           |
| Will FP64 keep declining on future GPUs?           | Likely yes on consumer / ML-class; HPC line uncertain (Rubin TBD).      |
| Does cuBLAS now do Ozaki natively?                 | Yes, on Blackwell, in CUDA 13.0 Update 2 (Oct 2025).                    |
| Do users still need to write Ozaki integration?    | Only for non-cuBLAS code, non-Blackwell GPUs, or precision-tuned cases. |
| Does `try-ozaki` still have value?                 | Yes — for source-level modernization, profiling, and validation.        |
| What should `try-ozaki` default to rewriting to?   | `cublasDgemm`. ozIMMU is now a fallback, not the primary target.        |

The strategic position to communicate to researchers: **the Ozaki Scheme is no longer the question. The question is whether your code is in a form that can use it.** That is what this tool helps determine.
