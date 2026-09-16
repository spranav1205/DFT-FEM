cmake_minimum_required(VERSION 3.13)

project(print_stiffness_matrix CXX)

find_package(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

add_executable(stiffness stiffness.cpp)
DEAL_II_SETUP_TARGET(stiffness)