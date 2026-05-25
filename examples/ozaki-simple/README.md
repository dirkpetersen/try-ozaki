# ozaki-simple

Minimal FP64 Fortran matrix-multiply benchmark used as a test case for `try-ozaki`.

## What it does

`matmul_fp64.f90` computes `C = A × B` for 2048×2048 double-precision matrices
using a hand-written triple-nested DO loop — the canonical pattern that the Ozaki
Scheme targets.  It runs 5 repetitions to accumulate ~3 minutes of wall time on
a single CPU core, making it long enough to observe real speedup when moved to
GPU Tensor Cores.

## Build & run locally

```bash
gfortran -O3 -o matmul_fp64 matmul_fp64.f90
./matmul_fp64
```

Expected output (times vary):

```
matmul_fp64: N=2048 FP64 triple-loop DGEMM benchmark
Running 5 repetitions...
------------------------------------------------------------
  Rep 1:   22.415 s    1.57 GFLOP/s
  C_ref(1,1) =  -2.345678E+01
  C_ref(N,N) =   1.234567E+01
  sum(C_ref) =   3.141592E+02
  Rep 2:   21.987 s    1.60 GFLOP/s   max_err= 0.000E+00
  ...
------------------------------------------------------------
matmul_fp64: done
```

## Run with try-ozaki

```bash
try-ozaki examples/ozaki-simple --no-submit   # analyze + rewrite only
try-ozaki examples/ozaki-simple               # full GPU pipeline
```

The analyzer detects the triple-nested FP64 DO loop and the rewriter replaces it
with a call to `OZAKI_DGEMM` from `ozaki_wrapper.f90`.
