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

#include "numerics/matrix_poisson_solver.h"
#include "numerics/matrix_free_poisson_solver.h"

// ---- Compile-time configuration ----
// Fallback to 3 if not specified via compiler flag (e.g., -DFE_DEGREE=3)
#ifndef FE_DEGREE
#  define FE_DEGREE 3
#endif
constexpr int fe_degree = FE_DEGREE;

// Toggle this macro via compiler flag or manually to swap backends:
// -DMATRIX_FREE=1 for Matrix-Free, -DMATRIX_FREE=0 for Assembled Matrix
#ifndef MATRIX_FREE
#  define MATRIX_FREE 0
#endif

#if MATRIX_FREE == 1
  using SolverType = Numerics::MatrixFreePoissonSolver<3, fe_degree>;
  const std::string method_name = "matrix-free";
#else
  using SolverType = Numerics::MatrixPoissonSolver<3>;
  const std::string method_name = "matrix-assembled";
#endif
// ------------------------------------

dealii::TrilinosWrappers::MPI::Vector
to_trilinos_vector(const dealii::TrilinosWrappers::MPI::Vector &v,
                    const dealii::IndexSet &,
                    MPI_Comm)
{
    return v;
}

dealii::TrilinosWrappers::MPI::Vector
to_trilinos_vector(const dealii::LinearAlgebra::distributed::Vector<double> &v,
                    const dealii::IndexSet &locally_owned_dofs,
                    MPI_Comm mpi_communicator)
{
    dealii::TrilinosWrappers::MPI::Vector out(locally_owned_dofs, mpi_communicator);
    for (const auto i : locally_owned_dofs)
        out[i] = v(i);
    out.compress(dealii::VectorOperation::insert);
    return out;
}

int main(int argc, char *argv[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        const unsigned int my_rank =
            dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        // Edit this list to test different mesh sizes within a single execution
        const std::vector<int> cells_per_edge_list = {24};

        const std::string output_dir = "benchmark_results";
        const std::string csv_path   = output_dir + "/performance_study.csv";

        if (my_rank == 0)
        {
            std::filesystem::create_directories(output_dir);
            
            // Check if file exists to determine if we write headers
            bool file_exists = std::filesystem::exists(csv_path);
            std::ofstream csv(csv_path, std::ios::app);
            if (!file_exists)
            {
                csv << "method,fe_degree,cells_per_edge,total_cells,dofs,"
                    << "setup_time_sec,assemble_time_sec,solve_time_sec,"
                    << "total_time_sec,cg_iterations,time_per_iter_sec,energy\n";
            }
        }

        for (const int n_cells_per_edge : cells_per_edge_list)
        {
            if (my_rank == 0)
            {
                std::cout << "\n=====================================\n"
                          << "Backend: " << method_name << " | FE Degree: " << fe_degree << "\n"
                          << "Running mesh with " << n_cells_per_edge << " cells per edge\n"
                          << "=====================================\n";
            }

            Physics::AtomSystem<3> atom_system;
            atom_system.add_atom(Physics::Atom<3>(dealii::Point<3>(0.0, 0.0, 0.0), 6.0, 0.7));
            atom_system.add_atom(Physics::Atom<3>(dealii::Point<3>(1.0, 1.0, 1.0), 6.0, 0.7));

            Physics::SmearedCharge<3>    forcing_term  = atom_system.get_charge_density();
            Physics::CoulombPotential<3> boundary_term = atom_system.get_boundary_potential();

            SolverType poisson_problem(fe_degree, atom_system); 

            // 1. Time Setup
            auto t0 = std::chrono::high_resolution_clock::now();
            poisson_problem.setup_system(n_cells_per_edge, -10.0, 10.0, boundary_term);
            auto t1 = std::chrono::high_resolution_clock::now();
            const double setup_time = std::chrono::duration<double>(t1 - t0).count();

            // 2. Time Assembly
            auto t2 = std::chrono::high_resolution_clock::now();
            poisson_problem.assemble_system(forcing_term);
            auto t3 = std::chrono::high_resolution_clock::now();
            const double assemble_time = std::chrono::duration<double>(t3 - t2).count();

            // 3. Time Solver
            auto t4 = std::chrono::high_resolution_clock::now();
            const int num_iterations = poisson_problem.solve(false);
            auto t5 = std::chrono::high_resolution_clock::now();
            const double solve_time = std::chrono::duration<double>(t5 - t4).count();

            // 4. Energy calculation
            dealii::TrilinosWrappers::MPI::Vector solution_trilinos =
                to_trilinos_vector(poisson_problem.get_solution(),
                                    poisson_problem.locally_owned_dofs,
                                    poisson_problem.mpi_communicator);

            const double energy = atom_system.electrostatic_energy(
                poisson_problem.dof_handler, poisson_problem.fe, solution_trilinos);

            // Metrics calculation
            const unsigned int total_cells = poisson_problem.triangulation.n_global_active_cells();
            const unsigned int total_dofs  = poisson_problem.dof_handler.n_dofs();
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
                          << "Energy            : " << std::setprecision(12) << energy << '\n';

                // Append to CSV
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
                    << std::setprecision(16) << energy << "\n";
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