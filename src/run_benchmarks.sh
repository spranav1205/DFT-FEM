#!/bin/bash
set -e

# Ensure this is run from the build directory
if [ ! -f "CMakeCache.txt" ]; then
    echo "Error: Please run this script from inside your 'build' folder!"
    echo "Example: cd build && ../run_benchmarks.sh"
    exit 1
fi

# Degrees to test: 2, 3, 4, 5
for degree in 5; do
  # Methods to test: 0 (Assembled Matrix), 1 (Matrix-Free)
  for mfree in 0; do
    
    echo "=========================================================="
    echo " CONFIGURING & COMPILING: FE_DEGREE=${degree} | MATRIX_FREE=${mfree}"
    echo "=========================================================="
    
    # Reconfigure CMake and compile cleanly inside the current build folder
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DFE_DEGREE=${degree} \
          -DMATRIX_FREE=${mfree} ..
          
    make -j$(nproc)

    echo "----------------------------------------------------------"
    echo " RUNNING..."
    echo "----------------------------------------------------------"
    
    # Run using MPI (adjust -np 4 to match your hardware cores)
    mpirun -np 8 ./dft_fem
    
  done
done

echo "Done! Data saved to build/benchmark_results/performance_study.csv"