!> matmul_fp64.f90
!> Dense FP64 matrix-matrix multiply benchmark.
!> Uses a triple-nested loop to compute C = A * B for N×N matrices.
!> Designed to run approximately 3 minutes on a single GPU/CPU core at N=2048.
!>
!> Build:
!>   gfortran -O3 -o matmul_fp64 matmul_fp64.f90 -lopenblas   ! (OpenBLAS DGEMM variant)
!>   gfortran -O3 -o matmul_fp64 matmul_fp64.f90               ! (pure loop variant)
!>
!> Run:
!>   ./matmul_fp64

program matmul_fp64
  implicit none

  integer, parameter :: dp = kind(0.d0)
  integer, parameter :: N  = 2048      ! matrix dimension; increase for longer run

  real(dp), allocatable :: A(:,:), B(:,:), C(:,:), C_ref(:,:)
  real(dp) :: t_start, t_end, elapsed
  real(dp) :: gflops, max_err
  integer  :: i, j, k, rep

  ! Number of repetitions — total wall time ≈ elapsed_one * NREP
  ! At N=2048, one loop pass takes ~20-40 s on a single CPU core.
  ! 5 reps → ~100-200 s ≈ 3 minutes.
  integer, parameter :: NREP = 5

  allocate(A(N,N), B(N,N), C(N,N), C_ref(N,N))

  ! Fill with reproducible pseudo-random values
  call init_matrix(A, N, seed=42)
  call init_matrix(B, N, seed=137)
  C_ref = 0.0_dp

  ! ── Compute reference (first rep) ──────────────────────────────────────────
  print '(a,i0,a)', "matmul_fp64: N=", N, " FP64 triple-loop DGEMM benchmark"
  print '(a,i0,a)', "Running ", NREP, " repetitions..."
  print '(a)', repeat("-", 60)

  call cpu_time(t_start)

  do j = 1, N
    do i = 1, N
      do k = 1, N
        C_ref(i,j) = C_ref(i,j) + A(i,k) * B(k,j)
      end do
    end do
  end do

  call cpu_time(t_end)
  elapsed = t_end - t_start
  gflops  = 2.0_dp * real(N,dp)**3 / (elapsed * 1.0e9_dp)

  print '(a,f8.3,a,f6.2,a)', "  Rep 1: ", elapsed, " s   ", gflops, " GFLOP/s"
  print '(a,e14.6)', "  C_ref(1,1) = ", C_ref(1,1)
  print '(a,e14.6)', "  C_ref(N,N) = ", C_ref(N,N)
  print '(a,e14.6)', "  sum(C_ref) = ", sum(C_ref)

  ! ── Additional repetitions ─────────────────────────────────────────────────
  do rep = 2, NREP
    C = 0.0_dp
    call cpu_time(t_start)

    do j = 1, N
      do i = 1, N
        do k = 1, N
          C(i,j) = C(i,j) + A(i,k) * B(k,j)
        end do
      end do
    end do

    call cpu_time(t_end)
    elapsed = t_end - t_start
    gflops  = 2.0_dp * real(N,dp)**3 / (elapsed * 1.0e9_dp)

    max_err = maxval(abs(C - C_ref))
    print '(a,i1,a,f8.3,a,f6.2,a,e10.3)', &
        "  Rep ", rep, ": ", elapsed, " s   ", gflops, " GFLOP/s   max_err=", max_err
  end do

  print '(a)', repeat("-", 60)
  print '(a)', "matmul_fp64: done"

  deallocate(A, B, C, C_ref)

contains

  subroutine init_matrix(M, n, seed)
    integer,  intent(in)  :: n, seed
    real(dp), intent(out) :: M(n,n)
    integer  :: i, j, s
    real(dp) :: x
    s = seed
    do j = 1, n
      do i = 1, n
        ! Simple LCG for reproducibility without random_number() state issues
        s = mod(s * 1664525 + 1013904223, 2147483647)
        x = real(s, dp) / 2147483647.0_dp
        M(i,j) = x - 0.5_dp   ! centre around zero
      end do
    end do
  end subroutine

end program matmul_fp64
