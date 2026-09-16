cmake_minimum_required(VERSION 3.13.4)

# 1. project() MUST come before find_package so CMake initializes the C++ compiler first
project(dft_fem LANGUAGES CXX)

# 2. Find deal.II (REQUIRED automatically handles missing package errors)
find_package(deal.II 9.7.1 REQUIRED
  HINTS ${DEAL_II_DIR} ${deal.II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# 3. Pull in deal.II's compiler flags, include dirs, definitions, build-type
#    handling (Debug/Release), etc. Must come after find_package(deal.II).
deal_ii_initialize_cached_variables()

# 4. Define the target.
set(TARGET "dft_fem")
set(TARGET_SRC poisson_401.cpp)

add_executable(${TARGET} ${TARGET_SRC})

# 5. Wire the target up to deal.II (include paths, libraries, MPI, LAPACK, etc.)
deal_ii_setup_target(${TARGET})