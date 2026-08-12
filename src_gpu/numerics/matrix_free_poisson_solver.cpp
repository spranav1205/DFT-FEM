#include "matrix_free_poisson_solver.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

namespace Numerics
{
    using namespace dealii;

    template <int dim, int fe_degree>
    MatrixFreePoissonSolver<dim, fe_degree>::MatrixFreePoissonSolver(
        int degree,
        const Physics::AtomSystem<dim> &atom_system)
        : PoissonSolverBase<dim>(degree)
        , atom_system(atom_system)
    {}

    template <int dim, int fe_degree>
    const LinearAlgebra::distributed::Vector<double> &
    MatrixFreePoissonSolver<dim, fe_degree>::get_solution_host() const
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
    void MatrixFreePoissonSolver<dim, fe_degree>::setup_system(
        int n_cells_per_edge,
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

        pcout << "Number of active CPU cells: " << triangulation.n_global_active_cells() << std::endl;
        dof_handler.distribute_dofs(fe);

        this->locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, this->locally_relevant_dofs);

        // Setup constraints and MatrixFree storage
        constraints.clear();
        constraints.reinit(this->locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
        constraints.close();
        pcout << "Number of constraints: " << constraints.n_constraints() << std::endl;

        // TODO 
        // Precomputes quantities: values of shape functions, gradients, JxW, etc. at quadrature points for each cell
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_partition;
        additional_data.mapping_update_flags =
            (update_values | update_gradients | update_JxW_values | update_quadrature_points);
        
        // Initialize MatrixFree storage for the system matrix and vectors
        // QGauss quadrature with fe_degree + 1 points for matrix elements
        // Higher order quadrature for the RHS to accurately integrate the forcing function
        mf_storage = std::make_shared<MatrixFree<dim, double>>();
        mf_storage->reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data); 

        system_matrix.initialize(mf_storage);
        system_matrix.compute_diagonal();

        mf_storage->initialize_dof_vector(solution);
        mf_storage->initialize_dof_vector(system_rhs);
        mf_storage->initialize_dof_vector(system_rhs_before);

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

        // QIterated: QGauss (m), n
        // m = number of quadrature points in each direction, n = number of intervals (subdivisions) in each direction
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
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(i) += f_q * fe_rhs.shape_value(i, q_index) * fe_rhs.JxW(q_index);
                }

                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
            }
        }

        // Sends buffered values to processes that own the DoFs
        // Recieve values from other processes for DoFs owned by this process
        system_rhs.compress(VectorOperation::add);

        // Store RHS before inhomogeneous boundary subtraction
        system_rhs_before = system_rhs;

        // u = u0 + g, where u0 is zero on the boundary 
        // A(u0 + g) = f => A(u0) = f - A(g)
        // Subtract inhomogeneous-boundary correction from RHS: f_mod = f - A(g) 
        LinearAlgebra::distributed::Vector<double> tmp;
        mf_storage->initialize_dof_vector(tmp);
        system_matrix.vmult_unconstrained(tmp, inhomogeneity_correction);
        system_rhs -= tmp;
    }

    template <int dim, int fe_degree>
    int MatrixFreePoissonSolver<dim, fe_degree>::solve(bool verbose)
    {
        SolverControl solver_control(2000, 1e-10);
        SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

        DiagonalMatrix<LinearAlgebra::distributed::Vector<double>> preconditioner;
        preconditioner.get_vector() = system_matrix.get_matrix_diagonal_inverse()->get_vector();

        solution = 0;
        solver.solve(system_matrix, solution, system_rhs, preconditioner);

        // u0 = u - g, where g is the inhomogeneous Dirichlet boundary values on constrained DOFs
        // soln = u0 + g
        solution += inhomogeneity_correction;
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
            this->pcout << "   Results written to partitioned processors matching: " << filename << std::endl;

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
    template class MatrixFreePoissonSolver<3, 1>;
    template class MatrixFreePoissonSolver<3, 2>;
    template class MatrixFreePoissonSolver<3, 3>;
    template class MatrixFreePoissonSolver<3, 4>;
    template class MatrixFreePoissonSolver<3, 5>;
}