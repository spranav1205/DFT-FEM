

set(TARGET "poisson_201")

set(TARGET_SRC
  ${TARGET}.cpp
)

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.9.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable DEAL_II_DIR that contains this path."
  )
endif()

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_KOKKOS)
  message(FATAL_ERROR
    "This program requires deal.II with MPI and Kokkos enabled.\n"
    "DEAL_II_WITH_MPI    = ${DEAL_II_WITH_MPI}\n"
    "DEAL_II_WITH_KOKKOS = ${DEAL_II_WITH_KOKKOS}\n"
  )
endif()

deal_ii_initialize_cached_variables()

project(${TARGET})

deal_ii_invoke_autopilot()
