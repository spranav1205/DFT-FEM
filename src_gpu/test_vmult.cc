// Standalone diagnostic: build one small mesh, set up both the CPU and GPU
// LaplaceOperator on it with identical constraints, apply vmult() once to
// the SAME input vector on each backend, and diff the outputs DOF-by-DOF.
//
// UPDATED: instead of just printing every DOF, we now:
//   1. Run on a SMALL mesh first (n_cells_per_edge=2) so a full per-DOF
//      table is still human-readable and any mismatch pattern is obvious
//      by eye, then optionally the large mesh.
//   2. Map every DOF to its physical (x,y,z) support point, so we can see
//      WHERE mismatches are (boundary layer? interior? a specific plane?)
//      instead of just an opaque index.
//   3. Only print DOFs that actually mismatch, plus a summary histogram of
//      mismatch coordinates, so the pattern is easy to read even on the
//      big mesh.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <iostream>
#include <map>

#include "numerics/laplace_operator.h"
#include "numerics/laplace_operator_gpu.h"

constexpr int dim       = 3;
constexpr int fe_degree = 1;

// Set to 1 for a small, fully-printable mesh; 0 for the original 20^3 mesh.
constexpr bool USE_SMALL_MESH   = true;
constexpr unsigned int N_CELLS_PER_EDGE = USE_SMALL_MESH ? 2 : 20;

int main(int argc, char *argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    Kokkos::print_configuration(std::cout, true);

    double max_abs_diff = 0.0; // read after the inner scope closes, for the
                                // final MATCH/MISMATCH message

    {   // ---- Inner scope: EVERY deal.II vector/operator lives and dies in
        // here. This matters because dealii::LinearAlgebra::distributed::
        // Vector is backed by a Kokkos::View even for MemorySpace::Host, not
        // just MemorySpace::Default -- so ALL of them (host and device) must
        // be destroyed before Kokkos::finalize() runs below, or their
        // destructors abort trying to deallocate through an already-torn-
        // down Kokkos runtime (exactly the abort you just hit).
        MPI_Comm mpi_communicator = MPI_COMM_WORLD;

        parallel::shared::Triangulation<dim> triangulation(mpi_communicator);
        GridGenerator::subdivided_hyper_cube(triangulation, N_CELLS_PER_EDGE, -1.0, 1.0);

        FE_Q<dim>       fe(fe_degree);
        DoFHandler<dim> dof_handler(triangulation);
        dof_handler.distribute_dofs(fe);

        IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        AffineConstraints<double> constraints;
        constraints.reinit(locally_relevant_dofs);
        constraints.close();

        std::cout << "DoFs: " << dof_handler.n_dofs()
                  << "  Cells: " << triangulation.n_active_cells()
                  << "  (n_cells_per_edge=" << N_CELLS_PER_EDGE << ")\n\n";

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

        LinearAlgebra::distributed::Vector<double> host_input(
            locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        for (const auto i : locally_owned_dofs)
            host_input(i) = std::sin(0.7 * static_cast<double>(i) + 0.3);
        host_input.update_ghost_values();

        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_partition;
        additional_data.mapping_update_flags =
            (update_values | update_gradients | update_JxW_values | update_quadrature_points);

        auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
        mf_storage->reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data);

        Numerics::LaplaceOperator<dim, fe_degree> cpu_operator;
        cpu_operator.initialize(mf_storage);

        LinearAlgebra::distributed::Vector<double> cpu_output;
        mf_storage->initialize_dof_vector(cpu_output);
        cpu_operator.vmult(cpu_output, host_input);

        Numerics::LaplaceOperatorGPU<dim, fe_degree> gpu_operator;
        gpu_operator.setup(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1));

        LinearAlgebra::distributed::Vector<double, MemorySpace::Default> device_input;
        gpu_operator.initialize_dof_vector(device_input);
        device_input.import_elements(host_input, VectorOperation::insert);

        LinearAlgebra::distributed::Vector<double, MemorySpace::Default> device_output;
        gpu_operator.initialize_dof_vector(device_output);
        gpu_operator.vmult(device_output, device_input);

        LinearAlgebra::distributed::Vector<double> gpu_output_host(
            locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        gpu_output_host.import_elements(device_output, VectorOperation::insert);

        std::cout << std::fixed << std::setprecision(6);

        unsigned int n_mismatches = 0;
        double min_abs_coord_of_mismatch = 1e300;
        double max_abs_coord_of_mismatch = -1e300;

        std::cout << "Mismatched DOFs (|diff| > 1e-8):\n";
        std::cout << std::setw(6) << "DOF" << std::setw(30) << "pos(x,y,z)"
                   << std::setw(14) << "CPU" << std::setw(14) << "GPU"
                   << std::setw(14) << "diff" << "\n";

        for (const auto i : locally_owned_dofs)
        {
            const double cpu_val = cpu_output(i);
            const double gpu_val = gpu_output_host(i);
            const double diff    = std::abs(cpu_val - gpu_val);
            max_abs_diff         = std::max(max_abs_diff, diff);

            if (diff > 1e-8)
            {
                ++n_mismatches;
                const auto &p = support_points.at(i);
                const double max_abs_p = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
                min_abs_coord_of_mismatch = std::min(min_abs_coord_of_mismatch, max_abs_p);
                max_abs_coord_of_mismatch = std::max(max_abs_coord_of_mismatch, max_abs_p);

                std::cout << std::setw(6) << i
                           << "  (" << std::setw(7) << p[0] << "," << std::setw(7) << p[1]
                           << "," << std::setw(7) << p[2] << ")"
                           << std::setw(14) << cpu_val << std::setw(14) << gpu_val
                           << std::setw(14) << diff << "\n";
            }
        }

        std::cout << "\nTotal DOFs: " << dof_handler.n_dofs()
                   << "  Mismatched: " << n_mismatches
                   << " (" << (100.0 * n_mismatches / dof_handler.n_dofs()) << "%)\n";
        std::cout << "Max |CPU - GPU| over all DOFs: " << max_abs_diff << "\n";

        if (n_mismatches > 0)
        {
            std::cout << "Mismatched-DOF coordinate range: max(|x|,|y|,|z|) in ["
                       << min_abs_coord_of_mismatch << ", " << max_abs_coord_of_mismatch << "]\n";
            std::cout << "  (domain half-extent is 1.0 -- values near 1.0 mean "
                          "mismatches are on/near the domain BOUNDARY; values "
                          "near 0 mean they're in the INTERIOR)\n";
        }
    }   // <-- every deal.II vector/operator above is destroyed HERE, before
        // Kokkos::finalize() runs. This is the actual fix for the abort.

    std::cout << (max_abs_diff < 1e-8 ? "MATCH (operators agree)"
                                       : "MISMATCH (operators disagree -- kernel bug confirmed)")
               << std::endl;

    // NOTE: do NOT call Kokkos::finalize() here. dealii::Utilities::MPI::
    // MPI_InitFinalize's destructor already calls it internally (via
    // dealii::InitFinalize::finalize()) when `mpi_initialization` goes out
    // of scope below. Calling it ourselves finalizes Kokkos twice, and the
    // second call aborts with "Kokkos::finalize() has already been called."
    // The inner scope above (all vectors/operators destroyed before this
    // point) was the actual fix needed -- deal.II handles the rest.

    return 0;
}