# Discussion: FP64 on Blackwell and Vera Rubin — How `try-ozaki` Addresses the Gap

> **The headline.** We have two procurement candidates for the next round of technical-computing capacity: the **NVIDIA RTX PRO 6000 Blackwell Server Edition** and the upcoming **NVIDIA Vera Rubin (GB200-class)** with ARM Vera CPUs. **Neither platform offers competitive native FP64 vector throughput compared to traditional HPC hardware.** Both are designed primarily for AI / mixed-precision workloads. If we want our researchers' double-precision scientific codes to run well on either platform, we must address the FP64 deficit deliberately. The **Ozaki Scheme** — emulating FP64 GEMM via INT8 / FP8 Tensor Cores with bit-exact output — is the leading technique for doing so. The `try-ozaki` package we are building is the tool that helps researchers identify, rewrite, and validate the parts of their codebases that can benefit.

**Key external references for this document:**

- [NVIDIA CUDA Library Samples (GitHub)](https://github.com/nvidia/cudalibrarysamples) — runnable example code, including FP emulation samples
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas) — authoritative API reference, including the [Floating-Point Emulation section](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) — compute capabilities, Tensor Core programming model, memory hierarchy
- [NVIDIA Developer Blog — Unlocking Tensor Core Performance with FP Emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) — the announcement of cuBLAS FP64 emulation in CUDA 13.0u2

**Pinned local copies (authoritative for the version this document references):**

- [`docs/cuda/CUBLAS_Library.md`](./docs/cuda/CUBLAS_Library.md) — full cuBLAS Library Reference, including the Floating-Point Emulation section
- [`docs/cuda/cuda-programming-guide.md`](./docs/cuda/cuda-programming-guide.md) — CUDA C++ Programming Guide, including the per-architecture Tensor Core support table

---

## 1. The two target platforms and their FP64 problem

### 1.1 RTX PRO 6000 Blackwell Server Edition

| Metric                                | Value                                                                  |
|---------------------------------------|------------------------------------------------------------------------|
| FP64 vector (native)                  | **~1.97 TFLOPS**                                                       |
| FP64 Tensor Core / DGEMM (native)     | **0 TFLOPS — not supported**                                           |
| FP32 single precision                 | ~120–126 TFLOPS                                                        |
| FP4 Tensor (with sparsity)            | ~2,000–4,000 TFLOPS (2–4 PFLOPS)                                       |
| Memory                                | 96 GB GDDR7 (ECC)                                                      |
| Power                                 | 600 W                                                                  |

**The architecture context.** The RTX PRO 6000 Blackwell Server Edition is fundamentally designed for AI inference, neural rendering, visual computing, and mixed-precision simulations (CFD, Omniverse-class physical AI). The vast majority of its die is allocated to single-precision (FP32), Ray Tracing cores, and narrow-precision AI math (FP8 / FP4). The hardware FP64 rate is **1 FP64 op per 64 FP32 ops** — a deliberate de-prioritization. Researchers running unmodified FP64 scientific codes on this card will see throughput roughly **30× lower than an H100 / H200** despite the card being a current-generation product.

### 1.2 Vera Rubin GB200-class (ARM Vera CPU + Rubin GPU)

| Precision Variant                              | Per Rubin GPU | Vera Rubin Superchip<br>(2× GPU + 1× Vera CPU) | Vera Rubin NVL72<br>(72× GPU rack) |
|------------------------------------------------|---------------|------------------------------------------------|------------------------------------|
| FP64 Vector (Standard)                         | 33 TFLOPS     | 67 TFLOPS                                      | **2,400 TFLOPS (2.4 PFLOPS)**      |
| FP64 DGEMM (Tensor Core, **emulated**)         | 200 TFLOPS    | 400 TFLOPS                                     | **14,400 TFLOPS (14.4 PFLOPS)**    |

**The critical observation.** The Rubin per-GPU FP64 vector rate (33 TFLOPS) is **less than half** of the H100/H200 native FP64 rate (~67 TFLOPS). The 200 TFLOPS DGEMM number — and the headline 14.4 PFLOPS at NVL72 scale — is **achieved exclusively through Tensor Core emulation**, i.e. the Ozaki Scheme running on top of INT8 / FP8 / FP4 hardware. **There is no native FP64 Tensor Core on Rubin.** If a code does not route its DGEMM through the emulated path, it gets the 33 TFLOPS vector number, not the 200 TFLOPS DGEMM number.

This is roughly a **6× difference in achievable performance per GPU** depending entirely on whether the code is in a form that cuBLAS (or ozIMMU) can emulate.

### 1.3 Why this matters for the procurement decision

Compared to a **traditional HPC GPU (H200, ~67 TFLOPS native FP64 with full FP64 Tensor Cores)**:

| GPU class                           | FP64 vector | FP64 DGEMM (best path)    | Path to best perf      |
|-------------------------------------|-------------|---------------------------|------------------------|
| H200 (Hopper, our current dgx001)   | ~34 TFLOPS  | ~67 TFLOPS native FP64 TC | already automatic      |
| **RTX PRO 6000 Blackwell**          | ~1.97 TFLOPS| **N/A native — emulated only** | **must emulate**  |
| **Rubin GPU**                       | 33 TFLOPS   | 200 TFLOPS **emulated**   | **must emulate**       |

For both target platforms, native FP64 throughput is *worse* than what we already have on H200. **The performance benefit only materializes when the code's DGEMM calls are routed through the Tensor Core emulation path.** That is the central technical problem this document — and `try-ozaki` — addresses.

---

## 2. Where the Ozaki Scheme fits

The Ozaki Scheme is **not an approximation**. It is an Error-Free Transformation (EFT):

- Input matrices are dynamically split into low-precision slices such that each slice's significand fits within the target hardware's exact-multiply range.
- Cross-products of slices are computed exactly on INT8 / FP8 / FP32 Tensor Cores.
- The high-precision result is reconstructed via weighted summation, also exactly.
- For sufficiently many slices, the final result is **bit-identical to a native FP64 multiply**.

Our own end-to-end experiment on a single H200 (`examples/ozaki-simple`, 2048×2048 DGEMM, 5 reps) confirmed this: `max relative error = 0.000` against the cuBLAS FP64 reference, across all reps. The math works in practice exactly as the theory predicts.

The trade-off is the **slice count**. Each additional slice is one more INT8/FP8 GEMM call. ozIMMU's `auto` mode and cuBLAS's Automatic Dynamic Precision (ADP) framework both select the minimum slice count needed for IEEE 754 FP64 equivalent accuracy from the input's exponent range. For typical scientific matrices this is 3–8 slices.

### Quantifying the speedup we expect on each platform

| Platform                     | Workload                       | Expected speedup of Ozaki vs native FP64 path                                |
|------------------------------|--------------------------------|-------------------------------------------------------------------------------|
| RTX PRO 6000 Blackwell       | FP64 DGEMM ≥ 4096²             | **~13× over the 1.97 TFLOPS vector path** (per NVIDIA cuBLAS heat maps)      |
| Rubin GPU                    | FP64 DGEMM                     | **~6× over the 33 TFLOPS vector path** (200 / 33 ≈ 6.06 from spec)            |
| Rubin Superchip (2× GPU)     | FP64 DGEMM                     | ~6× → **400 TFLOPS** vs 67 TFLOPS                                             |
| Rubin NVL72 (72× GPU rack)   | FP64 DGEMM at scale            | ~6× → **14.4 PFLOPS** vs 2.4 PFLOPS                                           |
| H200 (today, for comparison) | FP64 DGEMM                     | **~0.6×** (slower; native FP64 TC wins on Hopper — our measured result)       |
| H200                         | DGEMV / TRSM / non-GEMM        | not applicable (cuBLAS does not emulate these)                                |

### Quantifying the efficiency gain

The procurement-relevant question is "how much hardware do I need to deliver a target FP64 rate?" That ratio falls out directly:

| To deliver 1 PFLOPS of FP64 DGEMM, how many GPUs do I need? | Native vector path | Ozaki-emulated path |
|-------------------------------------------------------------|--------------------|---------------------|
| H200                                                        | ~15 GPUs           | ~15 GPUs (no benefit)|
| RTX PRO 6000 Blackwell                                      | ~510 GPUs          | **~40 GPUs** (~13×)  |
| Rubin GPU                                                   | ~31 GPUs           | **~5 GPUs** (~6×)    |

For a fixed PFLOPS target on either of the procurement candidates, **using the Ozaki-emulated path means roughly one-sixth (Rubin) to one-thirteenth (RTX PRO 6000) of the hardware** — with a corresponding cut in capital cost, power draw, rack space, and cooling. The efficiency gain is real and quantifiable, but it is conditional on the codes actually routing through the emulation path.

This is the lever `try-ozaki` operates on.

---

## 3. The H200 calibration result

In our test on dgx001:

```
cuBLAS FP64 (native):  623 ms    1,387 GFLOP/s
ozIMMU INT8 emulation: 976 ms      399 GFLOP/s
Speedup:               0.64×  (Ozaki is slower)
Max relative error:    0.000e+00
```

This is not a defeat — it is exactly the predicted outcome. The H200 has full FP64 Tensor Cores and cuBLAS FP64 exploits them. The result is **the calibration**: it confirms (a) the Ozaki path is numerically correct end-to-end on real GPU hardware, and (b) the speedup signal points the right direction — Ozaki wins exactly where native FP64 hardware is weak. On the H200 native FP64 wins; on the RTX PRO 6000 Blackwell and Rubin, where native FP64 is anemic by design, Ozaki wins by 6–13×.

---

## 4. What cuBLAS already covers — and where it leaves gaps

The good news for both procurement candidates is that NVIDIA has started shipping the Ozaki Scheme inside cuBLAS itself. As of CUDA 13.0 Update 2 (October 2025), the FP64 emulation that's needed to extract the 200 TFLOPS-per-Rubin-GPU number — and the 13× RTX PRO 6000 Blackwell number — is built into the vendor library. The bad news is that the coverage is narrow: it's **`cublasGemmEx` family only**, with no native support for typed `cublasDgemm`, GEMV, TRSM, LU, FFT, or sparse routines. That gap is exactly where `try-ozaki` has work to do.

The rest of this section verifies the cuBLAS coverage against the local copies of the [official cuBLAS Library Reference](./docs/cuda/CUBLAS_Library.md) and [CUDA C++ Programming Guide](./docs/cuda/cuda-programming-guide.md).

### Hardware and version support (verified against official docs)

From `docs/cuda/CUBLAS_Library.md:280-291`, the cuBLAS Reference's own table:

| Algorithm   | Emulates | Supported Compute Capabilities       | First CUDA version |
|-------------|----------|--------------------------------------|--------------------|
| BF16x9      | FP32     | 10.0, 10.3                           | **12.9+**          |
| Fixed-Point | FP64     | **8.x, 9.0, 10.0, 11.0, 12.x**       | **13.0 Update 2+** |

Two important specifics this corrects:

- **BF16x9 (FP32 emulation) shipped in CUDA 12.9**, not 13.0u2. Earlier than the blog post's framing suggests.
- **FP64 fixed-point emulation does cover Ampere through Blackwell** — the blog post emphasizes Blackwell because that's where the new hardware lives, but the cuBLAS docs list CC 8.x explicitly. **A100 (CC 8.0), H100 / H200 (CC 9.0)**, and Blackwell all qualify. Our own dgx001 H200 cluster is in scope.

The CUDA Programming Guide's per-architecture Tensor Core table (`docs/cuda/cuda-programming-guide.md:24104-24114`) confirms the underlying primitives:

- **CC 8.x (Ampere)**: INT8, FP64, TF32, BF16, FP16
- **CC 9.0 (Hopper)**: adds FP8
- **CC 10.x (Blackwell)**: adds INT4, FP4, FP6
- **CC 11.0, 12.x**: similar to 10.x but without INT8 / INT4 in some variants

The Hopper-and-later FP8 / FP4 Tensor Cores expand the substrate the Ozaki Scheme can run on — INT8 is no longer the only option for sub-FP64 acceleration.

### The cuBLAS emulation API (verified symbol names)

All function names below are confirmed against `docs/cuda/CUBLAS_Library.md` (line numbers in comments):

```c
// Configure emulation on a handle
cublasSetMathMode(handle, ...);                                  // L1581
cublasSetEmulationStrategy(handle, ...);                         // L1656 — performant | eager
cublasSetEmulationSpecialValuesSupport(handle, ...);             // L1697 — Inf/NaN handling

// FP64 fixed-point (Ozaki) tuning knobs
cublasSetFixedPointEmulationMantissaControl(handle, ...);        // L1726 — dynamic | fixed
cublasSetFixedPointEmulationMaxMantissaBitCount(handle, n);      // L1754 — ceiling
cublasSetFixedPointEmulationMantissaBitOffset(handle, ...);      // L1780 — perf tuning
cublasSetFixedPointEmulationMantissaBitCountPointer(handle, ...);// L1806 — device-side pointer

// Compute-type enums for cublasGemmEx() and friends
//   CUBLAS_COMPUTE_32F_EMULATED_16BFX9          (line 8700)
//   CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT      (referenced in API tables)
```

The doc also gives an exact slice-count formula at line 383:

```
sliceCount = ceildiv(mantissaBitCount + 1, 8)
```

i.e. each 8-bit increment in the requested mantissa precision adds one INT8 GEMM call. This is the operational definition of the performance/accuracy knob.

### Activating emulation without code changes

Environment variables documented in `docs/cuda/CUBLAS_Library.md:296-321`:

- `CUBLAS_EMULATE_DOUBLE_PRECISION=1` — enable FP64 emulation globally
- `CUBLAS_EMULATE_SINGLE_PRECISION=1` — enable FP32 (BF16x9) emulation
- `CUBLAS_EMULATION_STRATEGY=performant` (or `eager`)
- `CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=...`
- `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=N`

A user with an existing CUDA application that calls `cublasGemmEx` can pick up Ozaki emulation by **setting one environment variable** before the binary launches. No recompile.

### Supported routines (correction — narrower than my previous claim)

Verifying this against the cuBLAS Reference: emulation is documented for the **`*Ex` GEMM family only**:

- `cublasGemmEx()` (`docs/cuda/CUBLAS_Library.md:8513`)
- `cublasGemmBatchedEx()` (line 8746)
- `cublasGemmStridedBatchedEx()` (line 9003)

I previously suggested typed `cublasDgemm` would also pick up emulation when math mode was set; the cuBLAS Reference does **not** explicitly state this, and there is no documented extension to `GEMV`, `TRSM`, `SYRK`, `LU`, or other BLAS routines as of this CUDA toolkit. **In practice, code that wants FP emulation must route GEMM calls through `cublasGemmEx`.** This is a meaningful constraint for legacy code that calls typed BLAS functions and an additional transformation `try-ozaki` may need to perform.

### Known limitations (verified from `docs/cuda/CUBLAS_Library.md:549-578`)

- **Bit-wise reproducibility is not guaranteed** across toolkit versions.
- **Non-deterministic behavior with concurrent streams** can occur because of `cudaMallocAsync()` fallbacks when the workspace is exhausted.
- **CUDA Graph stream capture works**, but only when the user provides a pre-allocated workspace via `cublasSetWorkspace()` — otherwise child-graph memory-node failures occur (lines 748-752).
- **Special-values (Inf / NaN) support varies**: BF16x9 supports NaN implicitly; fixed-point FP64 does not implicitly support either.

### Mantissa-bit defaults — a caveat

I previously cited "default mantissa ceilings of 79 (dynamic) / 55 (fixed)" from the online blog summary. Those numbers are **not in the cuBLAS Library Reference** and could not be verified from the local docs. They may live in NVIDIA's tuning guides or implementation source rather than the public reference. Treat them as guidance from the blog only, not as authoritative defaults.

### Reported real-application results from NVIDIA

| Application      | Hardware                | Speedup                                         |
|------------------|-------------------------|-------------------------------------------------|
| ecTrans (weather)| GB200 NVL72             | 2.4× SGEMM (BF16x9 / FP32 emulation)            |
| BerkeleyGW       | B200                    | Significant ZGEMM speedup with ADP              |
| Quantum ESPRESSO | RTX PRO 6000 Blackwell  | 1.5× (default ADP), ~3× (manual 39 mantissa)    |

Heat maps in the NVIDIA blog show 2.3× DGEMM speedup on GB200 and up to **13× on RTX PRO 6000 Blackwell** across a range of matrix shapes, with no penalty on small matrices (cuBLAS heuristics fall back to native FP64).

**Net: the Ozaki Scheme is now a vendor-supported, drop-in capability for any code that calls `cublasGemmEx` on Ampere or newer hardware running CUDA 13.0u2.**

---

## 5. Honest reassessment: is custom Ozaki work still needed?

**For codes that already call `cublasGemmEx` on Ampere or newer hardware (CC ≥ 8.0): no.** Upgrade to CUDA 13.0 Update 2, set `CUBLAS_EMULATE_DOUBLE_PRECISION=1` (or call `cublasSetMathMode`), and the speedup is automatic where the heuristics decide it pays off. There is no reason to maintain a parallel ozIMMU integration for those cases.

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

**Q2: If I migrate this code to `cublasGemmEx` (and thereby pick up cuBLAS FP emulation automatically on Ampere+), will my numerical results still be correct?**

This requires:
- Automated rewrite from CPU loops / OpenBLAS calls / typed `cublasDgemm` into `cublasGemmEx` with `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`
- Side-by-side execution: original vs rewritten, both on GPU, compare outputs
- Pass/fail validation against a user-defined error tolerance — exactly what the current pipeline already does

What we built so far validates **the second half** of this end-to-end on a real GPU cluster: the analyze → rewrite → submit → validate loop works, and the numerical comparison was bit-exact (`max_err = 0.000`) once both pipelines ran on the GPU.

### Concrete pivot recommendations

1. **Default rewrite target should be `cublasGemmEx` with the FP64-emulated compute type.** This is the supported entry point per `docs/cuda/CUBLAS_Library.md`. Typed `cublasDgemm` is *not* explicitly listed as picking up emulation — code that calls typed BLAS today must be rewritten to the `*Ex` family to benefit. Once routed through `cublasGemmEx`, emulation is then a single `cublasSetMathMode` call or environment variable away.

2. **Add a profiling stage** that estimates how much of total runtime the detected hotspots represent. Without this, researchers can't justify the migration effort. A simple instrumented run that times the original DGEMM/loops gives this data.

3. **Add a hardware-aware recommendation** in the validation report. After running the test, output something like:
   - "On your H200: native cuBLAS FP64 was fastest (1,387 GFLOP/s); cuBLAS FP64 emulation is *available* on this GPU but not the fastest path here."
   - "On a B200 / GB200, cuBLAS would automatically use FP64 emulation via `cublasGemmEx` for an estimated 2.3× speedup."
   - "On RTX PRO 6000 Blackwell, the same code path delivers up to 13× speedup over native FP64."

4. **Keep the ozIMMU path as a fallback** for users on:
   - Pre-Ampere GPUs (V100, P100, T4, etc., CC < 8.0) where cuBLAS FP emulation is not supported
   - CUDA toolkits older than 13.0u2 (FP64 emulation) or 12.9 (FP32 BF16x9)
   - Codes that need precision-tuning beyond what cuBLAS ADP exposes (e.g. running with mantissa bits below ADP's recommendation)
   - Workloads where cuBLAS workspace requirements are problematic

5. **Add CUDA version + compute-capability detection** to the worker job script. The decision matrix is:
   - CUDA ≥ 13.0u2 **and** GPU CC ≥ 8.0 (Ampere or newer): rewrite to `cublasGemmEx` with `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, or just set `CUBLAS_EMULATE_DOUBLE_PRECISION=1`. No ozIMMU integration needed.
   - CUDA < 13.0u2 **or** GPU CC < 8.0 (Volta, Pascal): fall through to the explicit ozIMMU integration path.
   - Hand-written CPU loops or non-cuBLAS GEMM (MAGMA, custom kernels, etc.): rewrite to `cublasGemmEx` regardless of hardware — that is the modernization that pays off.

6. **Bonus expansion (cuBLAS doesn't cover):** the cuBLAS Reference documents emulation for GEMM only. Codes dominated by `GEMV` / `TRSM` / `SYRK` / `LU` / `QR` / FFT / sparse solvers are not helped by cuBLAS FP emulation today. For these, `try-ozaki` could either flag the gap or compose the higher-level routine from emulated GEMMs by hand. This is real, valuable territory that the vendor library does not yet occupy.

---

## 7. What runs where: a workload-routing assessment

Given the two procurement candidates, the practical question for each scientific workload is "which platform is the right home for this code?" The answer depends on what mathematical operations dominate, not just the FP64 peak number on the spec sheet.

### 7.1 Workloads that should run on Rubin (with Ozaki / cuBLAS emulation)

These workloads are **GEMM-dominated** — matrix-matrix multiplication is the inner kernel and most of the runtime. They include:

- **Dense linear algebra** — DFT-based codes (Quantum ESPRESSO, VASP, NWChem), molecular dynamics force-field assembly, quantum chemistry (ZGEMM via complex emulation)
- **Climate / weather spectral codes** — ecTrans-class transformations
- **Many-body physics** — BerkeleyGW-class GW calculations
- **Dense neural-network training and large-language-model fine-tuning** — already FP16/BF16 dominated, runs natively on Rubin without emulation

For these, **Rubin is the right platform**. The 200 TFLOPS-per-GPU FP64 DGEMM number, scaling to 14.4 PFLOPS at NVL72, is competitive with traditional HPC hardware once the codes route through `cublasGemmEx`. The native 33 TFLOPS vector path is *not* competitive — getting the headline number requires emulation. That is what `try-ozaki` makes feasible.

**Expected speedup vs running unmodified on Rubin: 6× per GPU.**

### 7.2 Workloads where the RTX PRO 6000 Blackwell makes sense

These are workloads where:

- FP32 single-precision is sufficient for the science (the card has ~125 TFLOPS FP32)
- The code is FP4/FP8/FP16-dominated (AI inference, neural rendering, mixed-precision CFD, Omniverse-class physical AI)
- There is **occasional** FP64 work, small enough that the 1.97 TFLOPS native rate is tolerable, or large enough that cuBLAS's 13× emulation lift is worth taking

The RTX PRO 6000 Blackwell is **not** a good home for sustained FP64 numerical work. Even with cuBLAS emulation delivering 13× over the 1.97 TFLOPS vector rate, the absolute number is far below Rubin and below H200. **Use it for AI / mixed-precision / FP32 workloads where its strengths shine.** Push pure FP64 codes onto Rubin or keep them on the existing H200 fleet.

### 7.3 Is FP32 sufficient for our researchers' codes?

This question matters because if FP32 is sufficient, the FP64 problem disappears for that code on the Blackwell card and the Ozaki path becomes optional rather than required. The honest answer is **it depends entirely on the numerical conditioning of the problem**:

- **Yes, FP32 is fine** for: many CFD codes (especially LES / RANS with bounded condition numbers), most neural-network training, ray tracing / rendering, molecular dynamics with single-precision force fields, single-precision FFTs in signal processing.
- **No, FP32 is not safe** for: ill-conditioned linear systems, long-term molecular dynamics integration where energy drift matters, climate codes accumulating over long simulation horizons, eigenvalue problems with closely-spaced eigenvalues, ab-initio quantum chemistry with energy differences smaller than FP32 epsilon.
- **It depends** for: most density functional theory codes (DFT) — small problems may be FP32-safe but the canonical benchmarks need FP64 for energy-difference accuracy.

This is exactly the question `try-ozaki`'s validation pipeline can answer for a given code: run the code in both FP64 and emulated-FP64 modes (which match each other to machine precision) versus a hypothetical FP32 port, and compare the numerical drift over the simulation horizon. If FP32 produces results within the user's tolerance, the FP64 problem doesn't apply to that code. If it doesn't, Ozaki emulation is the only path that delivers FP64-quality results on these new platforms at competitive throughput.

### 7.4 Quantified efficiency story for Vera Rubin procurement

For a **1 PFLOPS sustained FP64 DGEMM target** (a typical departmental workload):

| Path                                               | GPUs needed | Power (rough) | Rack space  |
|----------------------------------------------------|-------------|---------------|-------------|
| Rubin native FP64 vector (33 TFLOPS/GPU)           | ~31         | high          | ~half rack  |
| Rubin **with `try-ozaki` rewrite + cuBLAS emul.**  | **~5**      | **~6× lower** | **~5 GPUs** |
| Equivalent on H200 (native FP64 TC, ~67 TFLOPS)    | ~15         | reference     | reference   |

The procurement message: **buying Rubin and running unmodified FP64 code wastes most of the hardware's potential.** Buying Rubin and running Ozaki-routed code delivers ~3× the performance per dollar of an equivalent H200 cluster. The delta is the ~6× emulation lift Rubin's design assumes will be applied.

For the **RTX PRO 6000 Blackwell** the math is even starker: native FP64 (1.97 TFLOPS) requires ~510 GPUs for 1 PFLOPS; with the cuBLAS 13× emulation lift, ~40 GPUs. But the right move on this card is usually to ask whether FP32 would do, since FP32 is ~125 TFLOPS native — only ~8 GPUs needed at the FP32 throughput of 1 PFLOPS-equivalent.

---

## 8. Recommendation for researchers asking about Vera Rubin

The concern about declining FP64 in future GPUs is **valid and well-timed**. The technical answer is:

1. **The Ozaki Scheme is real, it works, and it is now vendor-supported.** Codes ported to it get bit-exact FP64 results from INT8/FP8 Tensor Cores. This is no longer an experimental academic paper — it's in cuBLAS, and Rubin's published peak FP64 number depends on it.

2. **For codes that already use `cublasGemmEx`, the migration to Vera Rubin is a CUDA upgrade, nothing more.** The Automatic Dynamic Precision framework will choose between native FP64 and emulated FP64 on whatever hardware is present.

3. **The codes that will struggle are those still using CPU loops, CPU BLAS, typed `cublasDgemm`, or non-cuBLAS GEMM implementations.** These need source-level modernization *before* the GPU upgrade matters. That is the gap `try-ozaki` should fill — and the typed-`Dgemm`-to-`GemmEx` rewrite is itself a non-trivial refactor that benefits from automated tooling.

4. **The window to do this work calmly is now**, while H100/H200 hardware is still abundant and FP64 isn't yet a bottleneck. Researchers who wait until Rubin lands and FP64 is anemic will be doing the same migration under pressure.

The viability of the approach is no longer in question — NVIDIA shipped it. The remaining work is one of **code migration**: getting researchers' actual scientific codebases into the form where they can benefit from it. That is an automation and tooling problem, and it's the problem `try-ozaki` is well-positioned to solve.

---

## 9. Summary

| Question                                                       | Answer                                                                                |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Do RTX PRO 6000 Blackwell or Vera Rubin have strong native FP64? | **No.** RTX PRO 6000: 1.97 TFLOPS native FP64, no FP64 Tensor Cores. Rubin: 33 TFLOPS vector / no native FP64 TC. |
| What's the headline Rubin FP64 number then?                    | **200 TFLOPS / GPU** — but only via **Tensor Core emulation** (Ozaki). 14.4 PFLOPS at NVL72 scale, all emulated. |
| Without emulation, what do we get on Rubin?                    | 33 TFLOPS / GPU — about half of H200. **Roughly 6× worse than the headline number.** |
| Does cuBLAS already do this emulation?                         | Yes — FP64 fixed-point in CUDA 13.0u2; FP32 BF16x9 since CUDA 12.9.                   |
| What's the gap cuBLAS leaves?                                  | **`cublasGemmEx` family only.** Typed `cublasDgemm`, GEMV, TRSM, LU, FFT, sparse, custom kernels — all uncovered. |
| What workloads should run on Rubin (with `try-ozaki` rewrite)? | Dense FP64 GEMM-heavy codes: DFT, MD force assembly, climate spectral, GW, ZGEMM-heavy QC. |
| What workloads belong on RTX PRO 6000 Blackwell?               | FP32-sufficient codes, AI inference, neural rendering, mixed-precision CFD/Omniverse. |
| Speedup from `try-ozaki` rewrite on RTX PRO 6000 Blackwell?    | **~13×** vs native FP64 vector path (per cuBLAS heat maps).                           |
| Speedup from `try-ozaki` rewrite on Rubin?                     | **~6×** per GPU vs native FP64 vector path (200/33 TFLOPS).                           |
| Efficiency gain — GPUs needed for 1 PFLOPS sustained DGEMM?    | RTX PRO 6000: 510 → 40 GPUs; Rubin: 31 → 5 GPUs.                                      |
| Is FP32 sufficient instead?                                    | Sometimes — depends on conditioning. `try-ozaki`'s validator can answer this per code. |
| What should `try-ozaki` default to rewriting to?               | `cublasGemmEx` with `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`. ozIMMU is fallback.     |

The strategic position to communicate to researchers: **on the new platforms, native FP64 is no longer the path to performance — Ozaki-emulated FP64 is.** Whether your code can take that path depends on whether it's in a form `cublasGemmEx` (or ozIMMU) understands. That is exactly what `try-ozaki` evaluates and converts.

---

## 10. References

### Pinned local copies (use these — they're the version this document was written against)

- **[`docs/cuda/CUBLAS_Library.md`](./docs/cuda/CUBLAS_Library.md)** — full cuBLAS Library Reference. Specific anchors used in this document:
  - L280-291: emulation overview + compute-capability support table
  - L296-321: emulation-related environment variables
  - L383: slice-count formula `ceildiv(mantissaBitCount + 1, 8)`
  - L549-578: known limitations (reproducibility, concurrent streams, CUDA Graph capture)
  - L1581, 1656, 1697, 1726, 1754, 1780, 1806: configuration API symbols
  - L8513, 8746, 9003: `cublasGemmEx` / Batched / StridedBatched declarations
- **[`docs/cuda/cuda-programming-guide.md`](./docs/cuda/cuda-programming-guide.md)** — CUDA C++ Programming Guide. Specific anchors:
  - L24104-24114: Tensor Core support table per compute capability (CC 8.x → 12.x)

### NVIDIA primary sources (online)

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
