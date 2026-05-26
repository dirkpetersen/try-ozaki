/* gpu_dgemm.cu — cuBLAS DGEMM backend for the FP64 baseline build.
 *
 * Provides a Fortran-callable DGEMM that runs on the GPU via cuBLAS.
 * Fortran passes all arguments by reference; this matches the standard
 * BLAS Fortran ABI (lowercase name with trailing underscore on Linux).
 *
 * The ozaki build replaces this with ozaki_wrapper.f90 + ozaki_wrapper.cpp
 * so both pipelines call the identical Fortran DGEMM interface.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_handle = nullptr;

static void ensure_handle() {
    if (!g_handle) {
        cublasStatus_t st = cublasCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[gpu_dgemm] cublasCreate failed: %d\n", st);
            exit(1);
        }
    }
}

/* Fortran DGEMM ABI: all args by pointer, name is lowercase + underscore */
extern "C" void dgemm_(
    const char *transa, const char *transb,
    const int  *m,  const int *n,  const int *k,
    const double *alpha,
    const double *A, const int *lda,
    const double *B, const int *ldb,
    const double *beta,
    double       *C, const int *ldc)
{
    ensure_handle();

    const int M = *m, N = *n, K = *k;
    const int LDA = *lda, LDB = *ldb, LDC = *ldc;

    /* Allocate device buffers */
    double *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)LDA * K * sizeof(double));
    cudaMalloc(&dB, (size_t)LDB * N * sizeof(double));
    cudaMalloc(&dC, (size_t)LDC * N * sizeof(double));

    cudaMemcpy(dA, A, (size_t)LDA * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, (size_t)LDB * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, (size_t)LDC * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasOperation_t opA = (*transa == 'N' || *transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (*transb == 'N' || *transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasDgemm(g_handle, opA, opB, M, N, K,
                alpha, dA, LDA, dB, LDB,
                beta,  dC, LDC);

    cudaMemcpy(C, dC, (size_t)LDC * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
