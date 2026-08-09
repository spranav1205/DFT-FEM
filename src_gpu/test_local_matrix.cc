// Single-cell local-matrix diagnostic: build a 1-cell mesh (fe_degree=1,
// dim=3 -> 8 DOFs), apply BOTH operators (unconstrained, so we see the raw
// kernel action with no boundary-condition interference) to each of the 8
// canonical basis vectors e_i, and print the resulting local 8x8 matrices
// side by side plus their difference.
//
// This tells us whether the CPU/GPU mismatch is:
//   (a) a permutation/DOF-numbering bug -- matrix is a scrambled version of
//       the correct one (rows/cols swapped), OR
//   (b) a magnitude/scaling bug -- right structure, wrong values (e.g.
//       missing/extra JxW factor, wrong quadrature weight), OR
//   (c) something structurally broken -- e.g. GPU matrix isn't symmetric
//       when it should be, wrong sparsity, garbage/zero entries.
//
// Uses vmult_unconstrained() on both sides so constraint handling
// (identical between backends or not) can't obscure the raw kernel output.

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

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

#include "numerics/laplace_operator.h"
#include "numerics/laplace_operator_gpu.h"

constexpr int dim       = 3;
constexpr int fe_degree = 1; // 8 DOFs on a single hex cell -- keep readable

int main(int argc, char *argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    // ---- Single cell ----
    parallel::shared::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::subdivided_hyper_cube(triangulation, /*n_cells_per_edge=*/1, -1.0, 1.0);

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const unsigned int n_dofs = dof_handler.n_dofs();
    std::cout << "DoFs: " << n_dofs << "  Cells: " << triangulation.n_active_cells() << "\n\n";

    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_relevant_dofs);
    constraints.close();

    // ---- CPU operator setup ----
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_partition;
    additional_data.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values | update_quadrature_points);

    auto mf_storage = std::make_shared<MatrixFree<dim, double>>();
    mf_storage->reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data);

    Numerics::LaplaceOperator<dim, fe_degree> cpu_operator;
    cpu_operator.initialize(mf_storage);

    // ---- GPU operator setup ----
    Numerics::LaplaceOperatorGPU<dim, fe_degree> gpu_operator;
    gpu_operator.setup(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1));

    // ---- Build local matrices column by column via e_i ----
    std::vector<std::vector<double>> cpu_matrix(n_dofs, std::vector<double>(n_dofs));
    std::vector<std::vector<double>> gpu_matrix(n_dofs, std::vector<double>(n_dofs));

    for (unsigned int i = 0; i < n_dofs; ++i)
    {
        // ---- CPU: apply to e_i ----
        LinearAlgebra::distributed::Vector<double> cpu_in, cpu_out;
        mf_storage->initialize_dof_vector(cpu_in);
        mf_storage->initialize_dof_vector(cpu_out);
        cpu_in = 0;
        if (locally_owned_dofs.is_element(i))
            cpu_in(i) = 1.0;
        cpu_in.update_ghost_values();

        cpu_operator.vmult_unconstrained(cpu_out, cpu_in);

        for (unsigned int j = 0; j < n_dofs; ++j)
            cpu_matrix[j][i] = cpu_out(j); // column i of the local matrix

        // ---- GPU: apply to e_i ----
        LinearAlgebra::distributed::Vector<double, MemorySpace::Default> gpu_in, gpu_out;
        gpu_operator.initialize_dof_vector(gpu_in);
        gpu_operator.initialize_dof_vector(gpu_out);

        LinearAlgebra::distributed::Vector<double> host_e_i(
            locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        host_e_i = 0;
        if (locally_owned_dofs.is_element(i))
            host_e_i(i) = 1.0;
        host_e_i.update_ghost_values();
        gpu_in.import_elements(host_e_i, VectorOperation::insert);

        gpu_operator.vmult_unconstrained(gpu_out, gpu_in);

        LinearAlgebra::distributed::Vector<double> host_out(
            locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        host_out.import_elements(gpu_out, VectorOperation::insert);

        for (unsigned int j = 0; j < n_dofs; ++j)
            gpu_matrix[j][i] = host_out(j);
    }

    // ---- Print CPU matrix ----
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU local matrix (A[row][col]):\n";
    for (unsigned int r = 0; r < n_dofs; ++r)
    {
        for (unsigned int c = 0; c < n_dofs; ++c)
            std::cout << std::setw(9) << cpu_matrix[r][c];
        std::cout << "\n";
    }

    std::cout << "\nGPU local matrix (A[row][col]):\n";
    for (unsigned int r = 0; r < n_dofs; ++r)
    {
        for (unsigned int c = 0; c < n_dofs; ++c)
            std::cout << std::setw(9) << gpu_matrix[r][c];
        std::cout << "\n";
    }

    std::cout << "\nDifference (CPU - GPU):\n";
    double max_diff = 0.0;
    for (unsigned int r = 0; r < n_dofs; ++r)
    {
        for (unsigned int c = 0; c < n_dofs; ++c)
        {
            const double d = cpu_matrix[r][c] - gpu_matrix[r][c];
            max_diff        = std::max(max_diff, std::abs(d));
            std::cout << std::setw(9) << d;
        }
        std::cout << "\n";
    }
    std::cout << "\nMax abs diff: " << max_diff << "\n";

    // ---- Quick structural checks ----
    double cpu_asym = 0.0, gpu_asym = 0.0;
    for (unsigned int r = 0; r < n_dofs; ++r)
        for (unsigned int c = 0; c < n_dofs; ++c)
        {
            cpu_asym = std::max(cpu_asym, std::abs(cpu_matrix[r][c] - cpu_matrix[c][r]));
            gpu_asym = std::max(gpu_asym, std::abs(gpu_matrix[r][c] - gpu_matrix[c][r]));
        }
    std::cout << "\nCPU symmetry violation (max |A_rc - A_cr|): " << cpu_asym << "\n";
    std::cout << "GPU symmetry violation (max |A_rc - A_cr|): " << gpu_asym << "\n";

    // Row sums should be ~0 for a pure Neumann Laplace local matrix
    // (constant function has zero gradient -> A * ones = 0)
    std::cout << "\nCPU row sums (should be ~0): ";
    for (unsigned int r = 0; r < n_dofs; ++r)
    {
        double s = 0;
        for (unsigned int c = 0; c < n_dofs; ++c) s += cpu_matrix[r][c];
        std::cout << std::setw(9) << s;
    }
    std::cout << "\nGPU row sums (should be ~0): ";
    for (unsigned int r = 0; r < n_dofs; ++r)
    {
        double s = 0;
        for (unsigned int c = 0; c < n_dofs; ++c) s += gpu_matrix[r][c];
        std::cout << std::setw(9) << s;
    }
    std::cout << "\n";

    return 0;
}