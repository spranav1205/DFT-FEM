#include <deal.II/base/mpi.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "physics/atom.h"
#include "physics/atom_system.h"
#include "physics/nuclear_charge_density.h"

/*
mkdir -p build_gpu && cd build_gpu
cmake -DDEAL_II_DIR=$HOME/software/dealii -DUSE_GPU=1 -DFE_DEGREE=3 ..
make -j

mkdir -p build_cpu && cd build_cpu
cmake -DDEAL_II_DIR=$HOME/software/dealii -DUSE_GPU=0 -DFE_DEGREE=3 ..
make -j
*/

// ---- Preprocessor Backend Configuration ----
#ifndef FE_DEGREE
#  define FE_DEGREE 5
#endif
constexpr int fe_degree = FE_DEGREE;

// Pass -DUSE_GPU=1 for GPU matrix-free, -DUSE_GPU=0 for CPU matrix-free
#ifndef USE_GPU
#  define USE_GPU 0
#endif

// #if USE_GPU == 1
// #  include "numerics/matrix_free_poisson_solver_gpu.h"
// using SolverType = Numerics::MatrixFreePoissonSolverGPU<3, fe_degree>;
// const std::string method_name = "gpu-matrix-free";
// #else
#  include "numerics/matrix_free_poisson_solver.h"
   using SolverType = Numerics::MatrixFreePoissonSolver<3, fe_degree>;
   const std::string method_name = "cpu-matrix-free";
// #endif


// --------------------------------------------

int main(int argc, char *argv[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        const unsigned int my_rank =
            dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        // Sweep from 24 to 32 cells per edge
        const std::vector<int> cells_per_edge_list = {24};

        const std::string output_dir = "benchmark_results";
        const std::string csv_path   = output_dir + "/performance_study.csv";

        if (my_rank == 0)
        {
            std::filesystem::create_directories(output_dir);
            
            // Append mode: create headers only if file doesn't exist
            bool file_exists = std::filesystem::exists(csv_path);
            std::ofstream csv(csv_path, std::ios::app);
            if (!file_exists)
            {
                csv << "method,fe_degree,cells_per_edge,total_cells,dofs,"
                    << "setup_time_sec,assemble_time_sec,solve_time_sec,"
                    << "total_time_sec,cg_iterations,time_per_iter_sec,energy\n"
                    << std::flush;
            }
        }

        for (const int n_cells_per_edge : cells_per_edge_list)
        {
            if (my_rank == 0)
            {
                std::cout << "\n=====================================\n"
                          << "Backend: " << method_name << " | FE Degree: " << fe_degree << "\n"
                          << "Running mesh with " << n_cells_per_edge << " cells per edge\n"
                          << "=====================================\n"
                          << std::flush;
            }

            Physics::AtomSystem<3> atom_system;
            atom_system.add_atom(Physics::Atom<3>(dealii::Point<3>(0.0, 0.0, 0.0), 12.0, 0.7));

            Physics::SmearedCharge<3>    forcing_term  = atom_system.get_charge_density();
            Physics::CoulombPotential<3> boundary_term = atom_system.get_boundary_potential();

            SolverType poisson_problem(fe_degree, atom_system); 

            // 1. Setup Phase
            auto t0 = std::chrono::high_resolution_clock::now();
            poisson_problem.setup_system(n_cells_per_edge, -10.0, 10.0, boundary_term);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double setup_time = std::chrono::duration<double>(t1 - t0).count();

            // 2. Assembly Phase
            auto t2 = std::chrono::high_resolution_clock::now();
            poisson_problem.assemble_system(forcing_term);
            auto t3 = std::chrono::high_resolution_clock::now();
            const double assemble_time = std::chrono::duration<double>(t3 - t2).count();

            // 3. Solve Phase
            auto t4 = std::chrono::high_resolution_clock::now();
            const int num_iterations = poisson_problem.solve(false);
            auto t5 = std::chrono::high_resolution_clock::now();
            const double solve_time = std::chrono::duration<double>(t5 - t4).count();

            // 4. Energy calculation
            const double energy = atom_system.electrostatic_energy(
                poisson_problem.get_dof_handler(),
                poisson_problem.get_fe(),
                poisson_problem.get_solution_host());

            // Metrics calculation
            const unsigned int total_cells = poisson_problem.get_triangulation().n_global_active_cells();
            const unsigned int total_dofs  = poisson_problem.get_dof_handler().n_dofs();
            const double total_time        = setup_time + assemble_time + solve_time;
            const double time_per_iter     = (num_iterations > 0) ? (solve_time / num_iterations) : 0.0;

            if (my_rank == 0)
            {
                std::cout << "Degrees of Freedom: " << total_dofs << '\n'
                          << "Setup time (s)    : " << setup_time << '\n'
                          << "Assemble time (s) : " << assemble_time << '\n'
                          << "Solve time (s)    : " << solve_time << '\n'
                          << "CG iterations     : " << num_iterations << '\n'
                          << "Time/Iteration (s): " << time_per_iter << '\n'
                          << "Energy            : " << std::setprecision(12) << energy << '\n'
                          << std::flush;

                // Open, append, and force immediate disk sync after each iteration
                std::ofstream csv(csv_path, std::ios::app);
                csv << method_name << ","
                    << fe_degree << ","
                    << n_cells_per_edge << ","
                    << total_cells << ","
                    << total_dofs << ","
                    << std::setprecision(6) << setup_time << ","
                    << assemble_time << ","
                    << solve_time << ","
                    << total_time << ","
                    << num_iterations << ","
                    << time_per_iter << ","
                    << std::setprecision(16) << energy << "\n"
                    << std::flush;
            }
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}