!> matmul_fp64.f90 — GPU DGEMM benchmark for try-ozaki
!>
!> Calls the standard DGEMM interface (BLAS Level 3).
!>
!>  Original build  → links gpu_dgemm.cu → cuBLAS DGEMM   (FP64 Tensor Cores / DGEMM units)
!>  Ozaki   build   → DGEMM replaced by OZAKI_DGEMM        (INT8 Tensor Cores via ozIMMU)
!>
!> Both pipelines run entirely on the GPU.
!>
!> The final RESULT: line is machine-parseable by the try-ozaki validator.

program matmul_fp64
  implicit none

  integer, parameter :: dp   = kind(0.d0)
  integer, parameter :: N    = 2048
  integer, parameter :: NREP = 5     ! timed repetitions (after 1 warm-up)

  real(dp), allocatable :: A(:,:), B(:,:), C(:,:), C_ref(:,:)
  real(dp) :: alpha, beta, t1, t2, elapsed, gflops, max_err
  real(dp) :: best_gflops
  integer  :: rep

  allocate(A(N,N), B(N,N), C(N,N), C_ref(N,N))

  alpha = 1.0_dp
  beta  = 0.0_dp

  call init_matrix(A, N, seed=42)
  call init_matrix(B, N, seed=137)

  print '(a,i0)', "matmul_fp64: N=", N
  print '(a)',    repeat("-", 60)

  ! ── Warm-up (not timed) ─────────────────────────────────────────────────
  C = 0.0_dp
  call DGEMM('N', 'N', N, N, N, alpha, A, N, B, N, beta, C, N)
  C_ref = C   ! save reference result from first call

  ! ── Timed repetitions ────────────────────────────────────────────────────
  best_gflops = 0.0_dp
  do rep = 1, NREP
    C = 0.0_dp
    call cpu_time(t1)
    call DGEMM('N', 'N', N, N, N, alpha, A, N, B, N, beta, C, N)
    call cpu_time(t2)

    elapsed = t2 - t1
    gflops  = 2.0_dp * real(N,dp)**3 / (elapsed * 1.0e9_dp)
    max_err = maxval(abs(C - C_ref))
    best_gflops = max(best_gflops, gflops)

    print '(a,i1,a,f8.3,a,f8.2,a,e10.3)', &
        "  Rep ", rep, ":  ", elapsed, " s   ", gflops, " GFLOP/s   max_err=", max_err
  end do

  print '(a)', repeat("-", 60)
  print '(a,f8.2,a)', "  Best: ", best_gflops, " GFLOP/s"

  ! ── Machine-parseable result line (validator reads ONLY this) ────────────
  print '(a,e22.15,a,e22.15,a,e22.15)', &
      "RESULT C11= ", C_ref(1,1), "  CNN= ", C_ref(N,N), "  sum= ", sum(C_ref)

  deallocate(A, B, C, C_ref)

contains

  subroutine init_matrix(M, n, seed)
    integer,  intent(in)  :: n, seed
    real(dp), intent(out) :: M(n,n)
    integer  :: i, j, s
    s = seed
    do j = 1, n
      do i = 1, n
        s = mod(s * 1664525 + 1013904223, 2147483647)
        M(i,j) = real(s, dp) / 2147483647.0_dp - 0.5_dp
      end do
    end do
  end subroutine

end program matmul_fp64
