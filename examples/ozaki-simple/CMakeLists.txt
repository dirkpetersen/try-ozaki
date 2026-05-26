cmake_minimum_required(VERSION 3.18)
project(ozaki_simple Fortran)

# CUDA is required for both builds (cuBLAS baseline + ozIMMU ozaki)
enable_language(CUDA CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 89 86 80)

add_executable(matmul_fp64 matmul_fp64.f90)

if(USE_OZAKI)
    # ── Ozaki build: replace DGEMM with ozIMMU INT8 Tensor Core emulation ──
    target_sources(matmul_fp64 PRIVATE
        ozaki_wrapper.f90    # defines module ozaki_wrapper + DGEMM interface
        ozaki_wrapper.cpp    # C++ shim calling mtk::ozimmu::gemm()
    )
    if(DEFINED OZIMMU_DIR)
        target_include_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/include")
        target_link_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/lib")
    endif()
    target_include_directories(matmul_fp64 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(matmul_fp64 PRIVATE ozimmu CUDA::cublas CUDA::cudart stdc++)
    target_compile_definitions(matmul_fp64 PRIVATE USE_OZAKI=1)
else()
    # ── Original build: cuBLAS DGEMM on the GPU ────────────────────────────
    target_sources(matmul_fp64 PRIVATE gpu_dgemm.cu)
    target_include_directories(matmul_fp64 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(matmul_fp64 PRIVATE CUDA::cublas CUDA::cudart)
endif()
