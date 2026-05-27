cmake_minimum_required(VERSION 3.18)
project(ozaki_simple Fortran)

# CUDA is required for both builds (cuBLAS baseline + ozIMMU/cuBLAS ozaki)
enable_language(CUDA CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_Fortran_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 89 86 80)

# Decide which Ozaki backend to use:
#   CUDA >= 13.2  →  cuBLAS native FP64 emulation  (preferred, no external dep)
#   CUDA <  13.2  →  ozIMMU library                 (fallback)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2")
    set(OZAKI_BACKEND "cublas" CACHE STRING "Ozaki backend (cublas|ozimmu)")
else()
    set(OZAKI_BACKEND "ozimmu" CACHE STRING "Ozaki backend (cublas|ozimmu)")
endif()
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "Ozaki backend selected: ${OZAKI_BACKEND}")

add_executable(matmul_fp64 matmul_fp64.f90)

if(USE_OZAKI)
    target_sources(matmul_fp64 PRIVATE ozaki_wrapper.f90)
    target_include_directories(matmul_fp64 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(matmul_fp64 PRIVATE USE_OZAKI=1)

    if(OZAKI_BACKEND STREQUAL "cublas")
        # ── Preferred: cuBLAS native FP64 emulation (CUDA >= 13.2) ─────────
        target_sources(matmul_fp64 PRIVATE ozaki_wrapper_cublas.cpp)
        target_link_libraries(matmul_fp64 PRIVATE CUDA::cublas CUDA::cudart stdc++)
    else()
        # ── Fallback: ozIMMU library (older CUDA) ──────────────────────────
        target_sources(matmul_fp64 PRIVATE ozaki_wrapper_ozimmu.cpp)
        if(DEFINED OZIMMU_DIR)
            target_include_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/include")
            target_link_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/lib")
        endif()
        target_link_libraries(matmul_fp64 PRIVATE ozimmu CUDA::cublas CUDA::cudart stdc++)
    endif()
else()
    # ── Baseline build: cuBLAS DGEMM on the GPU ────────────────────────────
    target_sources(matmul_fp64 PRIVATE gpu_dgemm.cu)
    target_include_directories(matmul_fp64 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(matmul_fp64 PRIVATE CUDA::cublas CUDA::cudart)
endif()
