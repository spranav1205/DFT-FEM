#include "matrix_free_poisson_solver_gpu.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "atom_system.h"

#include <fstream>

namespace Numerics
{
    using namespace dealii;

    template <int dim, int fe_degree>
    MatrixFreePoissonSolverGPU<dim, fe_degree>::MatrixFreePoissonSolverGPU(
        int degree, const Physics::AtomSystem<dim> &atom_system)
        : PoissonSolverBase<dim, VectorType>(degree)
        , atom_system(atom_system)
    {}

    template <int dim, int fe_degree>
    const typename MatrixFreePoissonSolverGPU<dim, fe_degree>::VectorType &
    MatrixFreePoissonSolverGPU<dim, fe_degree>::get_solution() const
    {
        return solution;
    }

    template <int dim, int fe_degree>
    const typename MatrixFreePoissonSolverGPU<dim, fe_degree>::VectorType &
    MatrixFreePoissonSolverGPU<dim, fe_degree>::get_rhs() const
    {
        return system_rhs_before;
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolverGPU<dim, fe_degree>::setup_system(
        int n_cells_per_edge, double start, double end,
        const Function<dim> &boundary_condition)
    {
        auto &triangulation = this->triangulation;
        auto &dof_handler   = this->dof_handler;
        auto &fe            = this->fe;
        auto &constraints   = this->constraints;
        auto &pcout         = this->pcout;

        GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);

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
                    if (refine) break;
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

        system_matrix.setup(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1));
        system_matrix.compute_inverse_diagonal(inverse_diagonal);

        system_matrix.initialize_dof_vector(solution);
        system_matrix.initialize_dof_vector(system_rhs);
        system_matrix.initialize_dof_vector(system_rhs_before);
        system_matrix.initialize_dof_vector(inhomogeneity_correction);

        LinearAlgebra::distributed::Vector<double> host_inhomogeneity(
            this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
        host_inhomogeneity = 0;
        constraints.distribute(host_inhomogeneity);
        inhomogeneity_correction.import_elements(host_inhomogeneity, VectorOperation::insert);
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolverGPU<dim, fe_degree>::assemble_system(const Function<dim> &forcing_function)
    {
        auto &dof_handler = this->dof_handler;
        auto &fe          = this->fe;
        auto &constraints = this->constraints;

        LinearAlgebra::distributed::Vector<double> host_rhs(
            this->locally_owned_dofs, this->mpi_communicator);
        host_rhs = 0;

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
                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, host_rhs);
            }
        }
        host_rhs.compress(VectorOperation::add);

        system_rhs.import_elements(host_rhs, VectorOperation::insert);

        VectorType tmp;
        system_matrix.initialize_dof_vector(tmp);
        system_matrix.vmult_unconstrained(tmp, inhomogeneity_correction);
        system_rhs -= tmp;

        system_rhs_before = system_rhs;
    }

    template <int dim, int fe_degree>
    int MatrixFreePoissonSolverGPU<dim, fe_degree>::solve(bool verbose)
    {
        SolverControl solver_control(20000, 1e-10);
        SolverCG<VectorType> solver(solver_control);

        DiagonalMatrix<VectorType> preconditioner;
        preconditioner.reinit(inverse_diagonal);

        solution = 0;
        solver.solve(system_matrix, solution, system_rhs, preconditioner);

        solution += inhomogeneity_correction;

        LinearAlgebra::distributed::Vector<double> host_solution(
            this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
        host_solution.import_elements(solution, VectorOperation::insert);
        this->constraints.distribute(host_solution);
        solution.import_elements(host_solution, VectorOperation::insert);

        if (verbose)
            this->pcout << "   " << solver_control.last_step()
                        << " CG iterations needed to obtain convergence." << std::endl;

        return solver_control.last_step();
    }

    template <int dim, int fe_degree>
    void MatrixFreePoissonSolverGPU<dim, fe_degree>::output_results(const std::string &filename, bool verbose) const
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
        output_solution.import_elements(solution, VectorOperation::insert);
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

    // Explicit template instantiations
    template class MatrixFreePoissonSolverGPU<3, 1>;
    template class MatrixFreePoissonSolverGPU<3, 2>;
    template class MatrixFreePoissonSolverGPU<3, 3>;
    template class MatrixFreePoissonSolverGPU<3, 4>;
    template class MatrixFreePoissonSolverGPU<3, 5>;
}