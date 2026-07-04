#include "matrix_poisson_solver.h"
#include "../physics/atom_system.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace Numerics
{
    using namespace dealii;

    template <int dim>
    MatrixPoissonSolver<dim>::MatrixPoissonSolver(int degree, const Physics::AtomSystem<dim> &atom_system)
        : PoissonSolverBase<dim, TrilinosWrappers::MPI::Vector>(degree), atom_system(atom_system)
    {}

    template <int dim>
    const TrilinosWrappers::MPI::Vector &MatrixPoissonSolver<dim>::get_solution() const
    {
        return solution;
    }

    template <int dim>
    const TrilinosWrappers::MPI::Vector &MatrixPoissonSolver<dim>::get_rhs() const
    {
        return system_rhs_before;
    }

    template <int dim>
    void MatrixPoissonSolver<dim>::setup_system(int n_cells_per_edge,
                  double start,
                  double end,
                  const Function<dim> &boundary_condition)
    {
        auto &triangulation = this->triangulation;
        auto &dof_handler   = this->dof_handler;
        auto &fe            = this->fe;
        auto &constraints   = this->constraints;
        auto &pcout         = this->pcout;

        GridGenerator::subdivided_hyper_cube(triangulation,
                                     n_cells_per_edge,
                                     start,
                                     end);

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

            const int total_marked =
                Utilities::MPI::sum(marked, this->mpi_communicator);

            pcout << "Marked " << total_marked
                << " cells for refinement." << std::endl;

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

        DynamicSparsityPattern dsp(this->locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, this->locally_owned_dofs,
                                                    this->mpi_communicator, this->locally_relevant_dofs);

        system_matrix.reinit(this->locally_owned_dofs, this->locally_owned_dofs, dsp, this->mpi_communicator);
        system_rhs.reinit(this->locally_owned_dofs, this->mpi_communicator);
        system_rhs_before.reinit(this->locally_owned_dofs, this->mpi_communicator);
        solution.reinit(this->locally_owned_dofs, this->mpi_communicator);
    }

    template <int dim>
    void MatrixPoissonSolver<dim>::assemble_system(const Function<dim> &forcing_function)
    {
        auto &dof_handler = this->dof_handler;
        auto &fe          = this->fe;
        auto &constraints = this->constraints;

        QGauss<dim> matrix_quad(fe.degree + 1);
        QIterated<dim> rhs_quad(QGauss<1>(6), 2);
        FEValues<dim> fe_matrix(fe, matrix_quad, update_values | update_gradients | update_JxW_values);
        FEValues<dim> fe_rhs(fe, rhs_quad, update_values | update_JxW_values | update_quadrature_points);
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs    = 0;

                fe_matrix.reinit(cell);
                fe_rhs.reinit(cell);

                const std::vector<Point<dim>> &quadrature_points_r = fe_rhs.get_quadrature_points();

                for (const unsigned int q_index : fe_matrix.quadrature_point_indices())
                    for (const unsigned int i : fe_matrix.dof_indices())
                        for (const unsigned int j : fe_matrix.dof_indices())
                            cell_matrix(i, j) += fe_matrix.shape_grad(i, q_index) *
                                                 fe_matrix.shape_grad(j, q_index) * fe_matrix.JxW(q_index);

                for (const unsigned int q_index : fe_rhs.quadrature_point_indices())
                {
                    const double f_q = forcing_function.value(quadrature_points_r[q_index]);
                    for (const unsigned int i : fe_rhs.dof_indices())
                        cell_rhs(i) += f_q * fe_rhs.shape_value(i, q_index) * fe_rhs.JxW(q_index);
                }

                cell->get_dof_indices(local_dof_indices);

                // Matrix+RHS overload: AffineConstraints uses cell_matrix to
                // fold the -A_local*g inhomogeneous-boundary correction into
                // the RHS automatically.
                constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                                        system_matrix, system_rhs);
            }
        }

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
        system_rhs_before = system_rhs;

        TrilinosWrappers::PreconditionJacobi::AdditionalData jacobi_data;
        jacobi_data.omega = 1.0;
        jacobi_preconditioner.initialize(system_matrix, jacobi_data);
    }

    template <int dim>
    int MatrixPoissonSolver<dim>::solve(bool verbose)
    {
        SolverControl solver_control(2000, 1e-10);
        TrilinosWrappers::SolverCG solver(solver_control);

        solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);
        this->constraints.distribute(solution);

        if (verbose)
            this->pcout << "   " << solver_control.last_step()
                        << " CG iterations needed to obtain convergence." << std::endl;

        return solver_control.last_step();
    }

    template <int dim>
    void MatrixPoissonSolver<dim>::output_results(const std::string &filename, bool verbose) const
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

        TrilinosWrappers::MPI::Vector output_solution;
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

    template class MatrixPoissonSolver<3>;
}