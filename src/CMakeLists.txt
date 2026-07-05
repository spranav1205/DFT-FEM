cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.3.0
  HINTS ${DEAL_II_DIR} ${deal.II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT deal.II_FOUND)
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "Pass -DDEAL_II_DIR=/path/to/deal.II or set the DEAL_II_DIR environment variable."
  )
endif()

deal_II_initialize_cached_variables()

project(dft_fem LANGUAGES CXX)

if(NOT DEAL_II_WITH_TRILINOS)
  message(FATAL_ERROR
    "*** deal.II must be built with Trilinos support. ***")
endif()

if(NOT DEAL_II_WITH_MPI)
  message(FATAL_ERROR
    "*** deal.II must be built with MPI support. ***")
endif()

# ---- Benchmark Configurations ----
# Set default values if not passed via the command line
if(NOT DEFINED FE_DEGREE)
  set(FE_DEGREE 3 CACHE STRING "Finite element polynomial degree")
endif()

if(NOT DEFINED MATRIX_FREE)
  set(MATRIX_FREE 0 CACHE STRING "Use Matrix-Free solver backend (1=True, 0=False)")
endif()

message(STATUS "Configuring dft_fem with FE_DEGREE=${FE_DEGREE}, MATRIX_FREE=${MATRIX_FREE}")
# ----------------------------------

# Automatically collect sources
file(GLOB PHYSICS_SRC CONFIGURE_DEPENDS
    physics/*.cpp)

file(GLOB NUMERICS_SRC CONFIGURE_DEPENDS
    numerics/*.cpp)

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

# Forward the CMake options into the C++ compiler preprocessing flags
target_compile_definitions(dft_fem PRIVATE
    FE_DEGREE=${FE_DEGREE}
    MATRIX_FREE=${MATRIX_FREE}
)

deal_II_setup_target(dft_fem)