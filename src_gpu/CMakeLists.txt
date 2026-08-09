# cmake_minimum_required(VERSION 3.13.4)

# find_package(deal.II 9.6.0 REQUIRED
#   HINTS /home/karikature/software/dealii-install ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
# )

# deal_ii_initialize_cached_variables()

# # Both CUDAToolkit_ROOT and CMAKE_CUDA_COMPILER must point at the SAME
# # toolkit (12.6) -- deal.II/Kokkos here were built against CUDA 12.6,
# # not the system-default 13.1, due to an upstream Kokkos/CUDA-13
# # incompatibility. Mixing them causes silent ABI/ptx mismatches.
# set(CUDAToolkit_ROOT "/usr/local/cuda-12.6")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
# set(CMAKE_CUDA_ARCHITECTURES "86")

# project(dft_fem LANGUAGES CXX CUDA)

# find_package(CUDAToolkit REQUIRED)

# # No Trilinos dependency: matrix_poisson_solver.{h,cpp} (the assembled-
# # matrix Trilinos path) is intentionally excluded below. Only the
# # Kokkos-based matrix-free CPU/GPU paths are built.

# if(NOT DEAL_II_WITH_MPI)
#   message(FATAL_ERROR
#     "*** deal.II must be built with MPI support. ***")
# endif()

# if(NOT DEFINED FE_DEGREE)
#   set(FE_DEGREE 3 CACHE STRING "Finite element polynomial degree")
# endif()
# message(STATUS "Configuring dft_fem with FE_DEGREE=${FE_DEGREE}")

# # ---- Shared physics sources (device-agnostic) ----
# file(GLOB PHYSICS_SRC CONFIGURE_DEPENDS physics/*.cpp)

# # ---- CPU matrix-free target ----
# add_executable(dft_fem_cpu
#   main.cpp
#   ${PHYSICS_SRC}
#   numerics/matrix_free_poisson_solver.cpp
#   numerics/laplace_operator.cpp
# )
# target_include_directories(dft_fem_cpu PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/physics
#   ${CMAKE_CURRENT_SOURCE_DIR}/numerics
# )
# target_compile_definitions(dft_fem_cpu PRIVATE
#   FE_DEGREE=${FE_DEGREE}
#   MATRIX_FREE=1
#   DFT_FEM_DEVICE_CPU=1
# )
# deal_ii_setup_target(dft_fem_cpu)

# # ---- GPU matrix-free target ----
# add_executable(dft_fem_gpu
#   main.cpp
#   ${PHYSICS_SRC}
#   numerics/matrix_free_poisson_solver_gpu.cpp
#   numerics/laplace_operator_gpu.cu
# )
# target_include_directories(dft_fem_gpu PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/physics
#   ${CMAKE_CURRENT_SOURCE_DIR}/numerics
# )
# target_compile_definitions(dft_fem_gpu PRIVATE
#   FE_DEGREE=${FE_DEGREE}
#   MATRIX_FREE=1
#   DFT_FEM_DEVICE_GPU=1
# )
# deal_ii_setup_target(dft_fem_gpu)
# set_property(TARGET dft_fem_gpu APPEND PROPERTY LINK_LIBRARIES CUDA::cudart)

cmake_minimum_required(VERSION 3.13.4)

# Pin CUDA 12.6 as early and forcefully as possible, before any find_package
# call (deal.II's included) gets a chance to resolve CUDAToolkit from PATH /
# a newer system install (e.g. CUDA 13.1) and cache that instead.
set(CUDAToolkit_ROOT "/usr/local/cuda-12.6" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)

find_package(deal.II 9.6.0 REQUIRED
    HINTS /home/karikature/software/dealii-install ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()

project(test_vmult LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Sanity check: fail loudly at configure time if CMake resolved a different
# CUDAToolkit than the one we pinned, instead of silently compiling against
# mismatched headers and producing confusing nvcc errors deep in Kokkos
# headers.
if(NOT CUDAToolkit_VERSION VERSION_EQUAL "12.6")
    message(WARNING
        "CUDAToolkit_ROOT was pinned to /usr/local/cuda-12.6 but CMake "
        "resolved CUDAToolkit_VERSION=${CUDAToolkit_VERSION} "
        "(CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}). "
        "A newer system CUDA install is likely shadowing the pinned one. "
        "Forcing include dirs explicitly below to compensate.")
endif()

add_executable(test_vmult
    test_vmult.cc
    numerics/laplace_operator.cpp
    numerics/laplace_operator_gpu.cu
)

target_include_directories(test_vmult PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/numerics
)

# THE ACTUAL FIX: explicitly prepend the 12.6 include dir as a SYSTEM
# BEFORE include, so it is searched ahead of whatever CMake's built-in CUDA
# language support and/or find_package(CUDAToolkit) auto-injected (which is
# apparently resolving to system CUDA 13.1's cuda_runtime_api.h, causing the
# cudaMemAdvise/cudaMemPrefetchAsync/cudaGraphAddDependencies signature
# mismatches against Kokkos, which was built against the 12.6 API).
target_include_directories(test_vmult SYSTEM BEFORE PRIVATE
    /usr/local/cuda-12.6/include
)

deal_ii_setup_target(test_vmult)

# Also strip any 13.x include dirs that may have been pulled in transitively
# by deal_ii_setup_target() / CUDA::cudart, and make sure our explicit 12.6
# dir stays first. This is defensive -- if the WARNING above didn't fire,
# this block is a no-op.
get_target_property(_tv_includes test_vmult INCLUDE_DIRECTORIES)
if(_tv_includes)
    list(REMOVE_ITEM _tv_includes "/usr/local/cuda/include")
    list(REMOVE_ITEM _tv_includes "/usr/local/cuda-13.1/include")
    set_property(TARGET test_vmult PROPERTY INCLUDE_DIRECTORIES
        /usr/local/cuda-12.6/include ${_tv_includes})
endif()

set_property(TARGET test_vmult APPEND PROPERTY LINK_LIBRARIES CUDA::cudart)