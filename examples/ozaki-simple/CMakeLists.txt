cmake_minimum_required(VERSION 3.16)
project(ozaki_simple Fortran)

set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")

option(USE_OZAKI "Link against ozIMMU for Ozaki DGEMM emulation" OFF)

add_executable(matmul_fp64 matmul_fp64.f90)

if(USE_OZAKI)
    enable_language(CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)

    # Find the CUDA toolkit so CMake knows where cuda_runtime.h lives
    find_package(CUDAToolkit REQUIRED)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ozaki_wrapper.f90")
        target_sources(matmul_fp64 PRIVATE ozaki_wrapper.f90)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ozaki_wrapper.cpp")
        target_sources(matmul_fp64 PRIVATE ozaki_wrapper.cpp)
    endif()
    if(DEFINED OZIMMU_DIR)
        target_include_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/include")
        target_link_directories(matmul_fp64 PRIVATE "${OZIMMU_DIR}/lib")
    endif()
    target_include_directories(matmul_fp64 PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(matmul_fp64 PRIVATE ozimmu CUDA::cublas CUDA::cudart stdc++)
    target_compile_definitions(matmul_fp64 PRIVATE USE_OZAKI=1)
endif()
