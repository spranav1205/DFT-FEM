#include "matrix_free_poisson_solver.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

// ================ MatrixFreePoissonSolver =================
// 
// Instead of loading an assembled matrix, this solver uses a matrix-free operator to apply
// Sparse matrices take up a lot of memory, especially for high-order finite elements. 
// Matrix-free methods avoid storing the matrix explicitly, and instead compute the action of the matrix on a vector on-the-fly.

namespace Numerics
{
    using namespace dealii;

    // ================= LaplaceOperator =================

    template <int dim, int fe_degree, typename number>
    LaplaceOperator<dim, fe_degree, number>::LaplaceOperator()
        : MatrixFreeOperators::Base<dim, VectorType>()
    {}

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::clear()
    {
        MatrixFreeOperators::Base<dim, VectorType>::clear();
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_apply(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const VectorType                            &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);

            // Read local coefficients u_j for this cell and hanging-node constraints.
            phi.read_dof_values(src);

            // Interpolate the local DoFs to compute ∇u_h at each quadrature point.
            phi.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
                // Store the flux F = ∇u_h at each quadrature point.
                phi.submit_gradient(phi.get_gradient(q), q);

            // Compute (A_local u_local)_i = ∫_K ∇φ_i · F dx.
            phi.integrate(EvaluationFlags::gradients);

            // Add the local operator result into the global vector.
            phi.distribute_local_to_global(dst);
        }
    }


    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::apply_add(VectorType &dst, const VectorType &src) const
    {
        this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_compute_diagonal(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const unsigned int                          &,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
        AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                    phi.submit_dof_value(VectorizedArray<number>(), j);
                phi.submit_dof_value(make_vectorized_array<number>(1.0), i);

                phi.evaluate(EvaluationFlags::gradients);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                    phi.submit_gradient(phi.get_gradient(q), q);
                phi.integrate(EvaluationFlags::gradients);

                diagonal[i] = phi.get_dof_value(i);
            }

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
                phi.submit_dof_value(diagonal[i], i);

            phi.distribute_local_to_global(dst);
        }
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
    {
        this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
        VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
        this->data->initialize_dof_vector(inverse_diagonal);

        unsigned int dummy = 0;
        this->data->cell_loop(&LaplaceOperator::local_compute_diagonal, this, inverse_diagonal, dummy);

        // Forces constrained (hanging-node/boundary) DOFs to have diagonal 1,
        // matching set_constrained_entries_to_one's role for the assembled
        // matrix: CG never needs to "solve" for these rows since their value
        // is fixed by constraints.distribute() afterward.
        this->set_constrained_entries_to_one(inverse_diagonal);

        for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
        {
            const double val = inverse_diagonal.local_element(i);
            inverse_diagonal.local_element(i) = (std::abs(val) > 1e-15) ? 1.0 / val : 1.0;
        }
    }

    // numerics/matrix_free_poisson_solver.cpp

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_apply_plain(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const VectorType                            &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);
            phi.read_dof_values_plain(src); // raw read: no constraint zeroing
            phi.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate(EvaluationFlags::gradients);
            phi.distribute_local_to_global(dst); // still respects constraints on write
        }
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::vmult_unconstrained(
        VectorType &dst, const VectorType &src) const
    {
        dst = 0;
        this->data->cell_loop(&LaplaceOperator::local_apply_plain, this, dst, src);
        dst.compress(VectorOperation::add);
    }

    // ================= MatrixFreePoissonSolver =================

    template <int dim, int fe_degree>
    MatrixFreePoissonSolver<dim, fe_degree>::MatrixFreePoissonSolver(int degree,
                                                                      const Physics::AtomSystem<dim> &atom_system)
        : PoissonSolverBase<dim, LinearAlgebra::distributed::Vector<double>>(degree)
        , atom_system(atom_system)
    {}

    template <int dim, int fe_degree>
    const LinearAlgebra::distributed::Vector<double> &
    MatrixFreePoissonSolver<dim, fe_degree>::get_solution() const
    {
        return solution;
    }

    template <int dim, int fe_degree>
    const LinearAlgebra::distributed::Vector<double> &
    MatrixFreePoissonSolver<dim, fe_degree>::get_rhs() const
    {
        return system_rhs_before;
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolver<dim, fe_degree>::setup_system(int n_cells_per_edge,
                                                                double start,
                                                                double end,
                                                                const Function<dim> &boundary_condition)
    {
        auto &triangulation = this->triangulation;
        auto &dof_handler   = this->dof_handler;
        auto &fe            = this->fe;
        auto &constraints   = this->constraints;
        auto &pcout         = this->pcout;

        GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);

        // Refine around each nucleus, identical logic to MatrixPoissonSolver:
        // a cell is marked if any of its vertices lies within
        // refinement_factor * r_c of any atom.
        const auto &atoms = this->atom_system.get_atoms();
        const double refinement_factor = 2.5;

        for (unsigned int cycle = 0; cycle < 3; ++cycle)
        {
            int marked = 0;

            for (const auto &cell : triangulation.active_cell_iterators())
            {
                if (!cell->is_locally_owned())
                    continue;

                bool refine = false;

                for (const auto &atom : atoms)
                {
                    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
                    {
                        if (cell->vertex(v).distance(atom.position) < refinement_factor * atom.r_c)
                        {
                            refine = true;
                            break;
                        }
                    }

                    if (refine)
                        break;
                }

                if (refine)
                {
                    cell->set_refine_flag();
                    ++marked;
                }
            }

            const int total_marked = Utilities::MPI::sum(marked, this->mpi_communicator);
            pcout << "Marked " << total_marked << " cells for refinement." << std::endl;
            triangulation.execute_coarsening_and_refinement();
        }

        pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
        dof_handler.distribute_dofs(fe);

        this->locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, this->locally_relevant_dofs);


        constraints.clear();
        constraints.reinit(this->locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
        constraints.close();
        pcout << "Number of constraints: " << constraints.n_constraints() << std::endl;

        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_partition;
        additional_data.mapping_update_flags =
            (update_values | update_gradients | update_JxW_values | update_quadrature_points);

        // reinit() with constraints: this ensures that the MatrixFree object knows about hanging-node and boundary constraints, so that it can handle them correctly during matrix-free operations.
        mf_storage = std::make_shared<MatrixFree<dim, double>>();
        mf_storage->reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data); 

        system_matrix.initialize(mf_storage);
        system_matrix.compute_diagonal();

        mf_storage->initialize_dof_vector(solution);
        mf_storage->initialize_dof_vector(system_rhs);
        mf_storage->initialize_dof_vector(system_rhs_before);

        // u0 is set on the boundary DOFs
        // We want to compute A(u0+g) where g is the value of the solution on the boundary. 
        // f = A(u) ---> A(u0) = f - A(g)
        mf_storage->initialize_dof_vector(inhomogeneity_correction);
        inhomogeneity_correction = 0;
        constraints.distribute(inhomogeneity_correction);
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolver<dim, fe_degree>::assemble_system(const Function<dim> &forcing_function)
    {
        auto &dof_handler = this->dof_handler;
        auto &fe          = this->fe;
        auto &constraints = this->constraints;

        system_rhs = 0;

        QIterated<dim> rhs_quad(QGauss<1>(6), 2);
        FEValues<dim> fe_rhs(fe, rhs_quad, update_values | update_JxW_values | update_quadrature_points);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                cell_rhs = 0;
                fe_rhs.reinit(cell);
                const std::vector<Point<dim>> &quadrature_points_r = fe_rhs.get_quadrature_points();

                for (const unsigned int q_index : fe_rhs.quadrature_point_indices())
                {
                    const double f_q = forcing_function.value(quadrature_points_r[q_index]);
                    for (const unsigned int i : fe_rhs.dof_indices())
                        cell_rhs(i) += f_q * fe_rhs.shape_value(i, q_index) * fe_rhs.JxW(q_index);
                }

                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
            }
        }
        // Compress the system_rhs vector to ensure that all contributions from different processes are summed correctly. 
        system_rhs.compress(VectorOperation::add);

        // Subtract the inhomogeneous-boundary correction from the RHS:
        // f = f - A(g) 
        LinearAlgebra::distributed::Vector<double> tmp;
        mf_storage->initialize_dof_vector(tmp);
        system_matrix.vmult_unconstrained(tmp, inhomogeneity_correction);
        system_rhs -= tmp;

        system_rhs_before = system_rhs;
    }

    template <int dim, int fe_degree>
    int MatrixFreePoissonSolver<dim, fe_degree>::solve(bool verbose)
    {
        SolverControl solver_control(2000, 1e-10);
        SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

        PreconditionJacobi<SystemMatrixType> preconditioner;
        preconditioner.initialize(system_matrix, 1.0);

        solution = 0;
        solver.solve(system_matrix, solution, system_rhs, preconditioner);

        solution += inhomogeneity_correction;

        // Distribute the solution to enforce constraints (hanging nodes, boundary conditions).
        this->constraints.distribute(solution);

        if (verbose)
            this->pcout << "   " << solver_control.last_step()
                        << " CG iterations needed to obtain convergence." << std::endl;

        return solver_control.last_step();
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolver<dim, fe_degree>::output_results(const std::string &filename, bool verbose) const
    {
        std::string partition_filename = filename;
        const unsigned int step = Utilities::MPI::this_mpi_process(this->mpi_communicator);
        const std::string step_string = "." + Utilities::int_to_string(step, 4);
        const std::string::size_type ext_pos = partition_filename.rfind(".vtu");
        if (ext_pos != std::string::npos)
            partition_filename.insert(ext_pos, step_string);
        else
            partition_filename += step_string + ".vtu";

        DataOut<dim> data_out;
        data_out.attach_dof_handler(this->dof_handler);

        LinearAlgebra::distributed::Vector<double> output_solution;
        output_solution.reinit(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
        output_solution = solution;
        output_solution.update_ghost_values();

        data_out.add_data_vector(output_solution, "potential");
        data_out.build_patches(this->fe.degree);

        std::ofstream output(partition_filename);
        data_out.write_vtu(output);

        if (verbose)
            this->pcout << "  Results written to partitioned processors matching: " << filename << std::endl;

        if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(this->mpi_communicator); ++i)
            {
                std::string p_file = filename;
                const std::string::size_type p_ext_pos = p_file.rfind(".vtu");
                if (p_ext_pos != std::string::npos)
                    p_file.insert(p_ext_pos, "." + Utilities::int_to_string(i, 4));
                else
                    p_file += "." + Utilities::int_to_string(i, 4) + ".vtu";
                filenames.push_back(p_file);
            }

            std::string master_filename = filename;
            if (ext_pos != std::string::npos)
                master_filename.replace(ext_pos, 4, ".pvtu");
            else
                master_filename += ".pvtu";

            std::ofstream master_output(master_filename);
            data_out.write_pvtu_record(master_output, filenames);
        }
    }

    // Explicit template instantiations for the supported dimensions and FE degrees.
    template class LaplaceOperator<3, 1, double>;
    template class LaplaceOperator<3, 2, double>;
    template class LaplaceOperator<3, 3, double>;
    template class LaplaceOperator<3, 4, double>;
    template class LaplaceOperator<3, 5, double>;
    template class MatrixFreePoissonSolver<3, 1>;
    template class MatrixFreePoissonSolver<3, 2>;
    template class MatrixFreePoissonSolver<3, 3>;
    template class MatrixFreePoissonSolver<3, 4>;
    template class MatrixFreePoissonSolver<3, 5>;
}