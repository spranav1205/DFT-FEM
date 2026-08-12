cmake_minimum_required(VERSION 3.13.4)

set(TARGET "dft_fem")

# Benchmark configuration options — defined early since they gate source
# selection and the deal.II feature checks below.
if(NOT DEFINED FE_DEGREE)
  set(FE_DEGREE 3 CACHE STRING "Finite element polynomial degree")
endif()

if(NOT DEFINED MATRIX_FREE)
  set(MATRIX_FREE 1 CACHE STRING "Use Matrix-Free solver backend (1=True, 0=False)")
endif()

if(NOT DEFINED USE_GPU)
  set(USE_GPU 0 CACHE STRING "Use GPU backend (1=GPU, 0=CPU)")
endif()

set(TARGET_SRC
    main.cpp
    physics/atom_system.cpp
    physics/nuclear_charge_density.cpp
    numerics/matrix_free_poisson_solver.cpp
    numerics/laplace_operator.cpp
)

# Only compile GPU units when USE_GPU is explicitly enabled
if(USE_GPU)
  list(APPEND TARGET_SRC
        numerics/matrix_free_poisson_solver_gpu.cpp
        numerics/laplace_operator_gpu.cpp
    )
endif()

# Find deal.II (Requires 9.9.0+ for Kokkos/GPU matrix-free features)
find_package(deal.II 9.9.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# MPI is always required; Kokkos is only required for the GPU backend.
if(NOT DEAL_II_WITH_MPI)
  message(FATAL_ERROR
    "\n*** deal.II must be built with MPI enabled. ***\n"
    "DEAL_II_WITH_MPI = ${DEAL_II_WITH_MPI}\n"
  )
endif()

if(USE_GPU AND NOT DEAL_II_WITH_KOKKOS)
  message(FATAL_ERROR
    "\n*** deal.II must be built with Kokkos enabled for GPU execution (USE_GPU=1). ***\n"
    "DEAL_II_WITH_KOKKOS = ${DEAL_II_WITH_KOKKOS}\n"
  )
endif()

deal_ii_initialize_cached_variables()

project(${TARGET} LANGUAGES CXX)

message(STATUS "Configuring ${TARGET} with FE_DEGREE=${FE_DEGREE}, MATRIX_FREE=${MATRIX_FREE}, USE_GPU=${USE_GPU}")

# Build Executable
add_executable(${TARGET} ${TARGET_SRC})

target_include_directories(${TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/physics
  ${CMAKE_CURRENT_SOURCE_DIR}/numerics
)

target_compile_definitions(${TARGET} PRIVATE
  FE_DEGREE=${FE_DEGREE}
  MATRIX_FREE=${MATRIX_FREE}
  USE_GPU=${USE_GPU}
)

# Attach deal.II compiler/GPU flags and libraries
deal_ii_setup_target(${TARGET})