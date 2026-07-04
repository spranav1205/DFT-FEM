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
// #include "numerics/matrix_free_poisson_solver.h"

// Fixed FE degree used by the matrix-free operator (compile-time constant).
constexpr int fe_degree = 4;

// ---- Backend switch: change this ONE line to swap solvers. ----
// Matrix-free (fast, low memory, needs fe_degree fixed at compile time):
// using SolverType = Numerics::MatrixFreePoissonSolver<3, fe_degree>;
// Trilinos-assembled (reference/debug backend):
using SolverType = Numerics::MatrixPoissonSolver<3>;
// -----------------------------------------------------------------

// Helper: build a Trilinos vector view of the solution for AtomSystem's
// energy computation, regardless of which backend's native VectorType
// was produced. No-op copy for the Trilinos backend, element copy for
// the matrix-free backend.
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

        const std::vector<int> cells_per_edge_list = {12};

        if (my_rank == 0)
            std::filesystem::create_directories("convergence parallel");

        std::ofstream csv;
        if (my_rank == 0)
        {
            csv.open("convergence parallel/energy_convergence.csv");
            csv << "cells_per_edge,total_cells,cg_iterations,solve_time_sec,energy\n";
        }

        for (const int n_cells_per_edge : cells_per_edge_list)
        {
            if (my_rank == 0)
                std::cout << "\n=====================================\n"
                          << "Running mesh with " << n_cells_per_edge << " cells per edge\n"
                          << "=====================================\n";

            // ---- Physics: define the atom system ----
            Physics::AtomSystem<3> atom_system;
            atom_system.add_atom(Physics::Atom<3>(dealii::Point<3>(0.0, 0.0, 0.0),
                                                   /*charge=*/6.0,
                                                   /*r_c=*/0.7));
            atom_system.add_atom(Physics::Atom<3>(dealii::Point<3>(1.0, 1.0, 1.0),
                                                   /*charge=*/6.0,
                                                   /*r_c=*/0.7));

            Physics::SmearedCharge<3>    forcing_term  = atom_system.get_charge_density();
            Physics::CoulombPotential<3> boundary_term = atom_system.get_boundary_potential();

            // ---- Numerics: solve (backend chosen by SolverType above) ----
            SolverType poisson_problem(fe_degree, atom_system); 

            poisson_problem.setup_system(n_cells_per_edge, -10.0, 10.0, boundary_term);
            poisson_problem.assemble_system(forcing_term);

            auto start_time = std::chrono::high_resolution_clock::now();
            const int num_iterations = poisson_problem.solve(false);
            auto end_time = std::chrono::high_resolution_clock::now();
            const double elapsed_time =
                std::chrono::duration<double>(end_time - start_time).count();

            dealii::TrilinosWrappers::MPI::Vector solution_trilinos =
                to_trilinos_vector(poisson_problem.get_solution(),
                                    poisson_problem.locally_owned_dofs,
                                    poisson_problem.mpi_communicator);

            const double energy = atom_system.electrostatic_energy(
                poisson_problem.dof_handler, poisson_problem.fe, solution_trilinos);

            const unsigned int total_cells = poisson_problem.triangulation.n_global_active_cells();

            {
                std::string filename = "convergence parallel/solution_" +
                                        std::to_string(n_cells_per_edge) + ".vtu";
                poisson_problem.output_results(filename, true);
            }

            if (my_rank == 0)
            {
                std::cout << "Cells/edge      : " << n_cells_per_edge << '\n'
                          << "Total cells     : " << total_cells << '\n'
                          << "CG iterations   : " << num_iterations << '\n'
                          << "Solve time (s)  : " << elapsed_time << '\n'
                          << "Energy          : " << std::setprecision(12) << energy << '\n';

                csv << n_cells_per_edge << "," << total_cells << "," << num_iterations << ","
                    << std::setprecision(16) << elapsed_time << "," << energy << "\n";
                csv.flush();
            }
        }

        if (my_rank == 0)
        {
            csv.close();
            std::cout << "\nConvergence study complete.\n"
                      << "Results saved to:\n"
                      << "  convergence parallel/energy_convergence.csv\n";
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}