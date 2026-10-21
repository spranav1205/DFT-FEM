
cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.4.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()

project(bench_schwarz LANGUAGES CXX)

set(FE_DEGREE 4 CACHE STRING "FE_Q polynomial degree")

option(BENCH_NATIVE "Compile with -march=native" ON)
option(BENCH_VEC_REPORT "Enable GCC vectorization diagnostics" OFF)

add_executable(bench bench_schwarz.cpp)

target_compile_definitions(bench PRIVATE FE_DEGREE=${FE_DEGREE})

target_compile_options(bench PRIVATE
  -O3
  -fno-math-errno
  -g
  -fno-omit-frame-pointer
)

if(BENCH_NATIVE)
  target_compile_options(bench PRIVATE -march=native)
endif()

if(BENCH_VEC_REPORT)
  target_compile_options(bench PRIVATE
  -O3
  -march=native
  -fopenmp-simd
)
endif()

deal_ii_setup_target(bench)
