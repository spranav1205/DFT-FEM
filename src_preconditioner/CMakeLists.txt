cmake_minimum_required(VERSION 3.13.4)

# 1. project() MUST come before find_package so CMake initializes the C++ compiler first
project(dft_fem LANGUAGES CXX)

# 2. Find deal.II (REQUIRED automatically handles missing package errors)
find_package(deal.II 9.7.1 REQUIRED
  HINTS ${DEAL_II_DIR} ${deal.II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# 3. Initialize deal.II compiler flags and variables
deal_II_initialize_cached_variables()

# 4. Feature checks
# IMPORTANT: If you recently uninstalled Trilinos, comment out this check!
if(NOT DEAL_II_WITH_TRILINOS)
  message(FATAL_ERROR
    "*** deal.II must be built with Trilinos support. ***")
endif()

if(NOT DEAL_II_WITH_MPI)
  message(FATAL_ERROR
    "*** deal.II must be built with MPI support. ***")
endif()

# ---- Benchmark Configurations ----
if(NOT DEFINED FE_DEGREE)
  set(FE_DEGREE 3 CACHE STRING "Finite element polynomial degree")
endif()

if(NOT DEFINED MATRIX_FREE)
  set(MATRIX_FREE 0 CACHE STRING "Use Matrix-Free solver backend (1=True, 0=False)")
endif()

message(STATUS "Configuring dft_fem with FE_DEGREE=${FE_DEGREE}, MATRIX_FREE=${MATRIX_FREE}")
# ----------------------------------

# Collect source files
file(GLOB PHYSICS_SRC CONFIGURE_DEPENDS physics/*.cpp)
file(GLOB NUMERICS_SRC CONFIGURE_DEPENDS numerics/*.cpp)

add_executable(dft_fem
    main.cpp
    ${PHYSICS_SRC}
    ${NUMERICS_SRC}
)

target_include_directories(dft_fem PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/physics
    ${CMAKE_CURRENT_SOURCE_DIR}/numerics
)

# Pass preprocessor definitions
target_compile_definitions(dft_fem PRIVATE
    FE_DEGREE=${FE_DEGREE}
    MATRIX_FREE=${MATRIX_FREE}
)

# Attach deal.II headers, build flags, and link targets
deal_II_setup_target(dft_fem)