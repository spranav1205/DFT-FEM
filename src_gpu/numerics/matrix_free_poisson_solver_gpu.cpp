#include "matrix_free_poisson_solver_gpu.h"

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
    MatrixFreePoissonSolverGPU<dim, fe_degree>::MatrixFreePoissonSolverGPU(
        int degree, const Physics::AtomSystem<dim> &atom_system)
        : PoissonSolverBase<dim>(degree), // Explicitly initialize base class
        atom_system(atom_system)
    {}
    template <int dim, int fe_degree>
    const LinearAlgebra::distributed::Vector<double> &
    MatrixFreePoissonSolverGPU<dim, fe_degree>::get_solution_host() const
    {
        return this->ghost_solution_host;
    }

    template <int dim, int fe_degree>
    const LinearAlgebra::distributed::Vector<double> &
    MatrixFreePoissonSolverGPU<dim, fe_degree>::get_rhs() const
    {
        return this->system_rhs_host;
    }

        template <int dim, int fe_degree>
    void MatrixFreePoissonSolverGPU<dim, fe_degree>::setup_system(
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
 
        pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
        
        dof_handler.distribute_dofs(fe);
        
        this->locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, this->locally_relevant_dofs);
 
        // Set up constraints and MatrixFree storage
        constraints.clear();
        constraints.reinit(this->locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
        constraints.close();
 
        pcout << "Number of constraints: " << constraints.n_constraints() << std::endl;
 
        // Resets GPU memory for the system matrix
        this->system_matrix_dev.reset(new SystemMatrixType(dof_handler, constraints));
 
        // Compute the diagonal of the operator now, so that
        // get_matrix_diagonal_inverse() (used by the CG preconditioner in solve())
        // returns a valid, already-initialized vector instead of a null shared_ptr.
        // This must be recomputed here since it depends on the current mesh/constraints.
        this->system_matrix_dev->compute_diagonal();
 
        // 1. Create host vector for boundary values and apply constraints
        LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host> 
            boundary_values_host(this->locally_owned_dofs, 
                                 this->locally_relevant_dofs, 
                                 this->mpi_communicator);
        boundary_values_host = 0.0;
        constraints.distribute(boundary_values_host);
 
        // 2. Initialize device vector for boundary values and transfer host values to GPU
        this->system_matrix_dev->initialize_dof_vector(this->boundary_values_dev);
        LinearAlgebra::ReadWriteVector<double> rw_bv(this->locally_owned_dofs);
        rw_bv.import_elements(boundary_values_host, VectorOperation::insert);
        this->boundary_values_dev.import_elements(rw_bv, VectorOperation::insert);
 
        // 3. Initialize host solution vector with ghost elements
        this->ghost_solution_host.reinit(this->locally_owned_dofs,
                                         this->locally_relevant_dofs,
                                         this->mpi_communicator);
            
        // 4. Initialize device vectors for solution and RHS
        this->system_matrix_dev->initialize_dof_vector(this->solution_dev);
        this->system_rhs_dev.reinit(this->solution_dev);
 
        // Set initial guess for the solution to the boundary values
        this->solution_dev = this->boundary_values_dev;
    }

    
    template <int dim, int fe_degree>
    void MatrixFreePoissonSolverGPU<dim, fe_degree>::assemble_system(const Function<dim> &forcing_function)
    {
        // Reinitialise host RHS vector stored as a member variable
        this->system_rhs_host.reinit(this->locally_owned_dofs, 
                                     this->locally_relevant_dofs, 
                                     this->mpi_communicator);
        this->system_rhs_host = 0.0;

        const QIterated<dim> quadrature_formula(QGauss<1>(6), 3);
        FEValues<dim> fe_values(this->fe,
                                quadrature_formula,
                                update_values | update_quadrature_points |
                                    update_JxW_values);

        const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
        const unsigned int n_q_points    = quadrature_formula.size();

        Vector<double>                       cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const Physics::SmearedCharge<dim> rho(this->atom_system.get_atoms());

        for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                cell_rhs = 0;
                fe_values.reinit(cell);

                for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
                {
                    const double rho_q =
                        rho.value(fe_values.quadrature_point(q_index));

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        cell_rhs(i) += fe_values.shape_value(i, q_index) * rho_q *
                                       fe_values.JxW(q_index);
                }

                cell->get_dof_indices(local_dof_indices);
                this->constraints.distribute_local_to_global(cell_rhs,
                                                             local_dof_indices,
                                                             this->system_rhs_host);
            }
        }
        this->system_rhs_host.compress(VectorOperation::add);

        // Transfer host RHS vector to device RHS vector
        LinearAlgebra::ReadWriteVector<double> rw_vector(this->locally_owned_dofs);
        rw_vector.import_elements(this->system_rhs_host, VectorOperation::insert);
        this->system_rhs_dev.import_elements(rw_vector, VectorOperation::insert);

        // Enforce boundary values on constrained DOFs of system_rhs_dev
        this->system_matrix_dev->copy_constrained_values(this->boundary_values_dev,
                                                         this->system_rhs_dev);

        // V IMPORTANT: The initial guess for solution_dev is set to boundary_values_dev (x_i = g_i).
        // Since A_ii = 1.0 on constrained DoFs, the initial residual (r_i = b_i - A_ii * x_i) 
        // is 0.0 at iteration 0, preventing the solver from altering the boundary values.

        // Instead of solving for u0 = u - g, then adding g back to the solution
        // we directly solve for u, with the initial guess set to the boundary values g.
    }   

    template <int dim, int fe_degree>
    int MatrixFreePoissonSolverGPU<dim, fe_degree>::solve(bool verbose)
    {
        SolverControl solver_control(2000, 1e-10);
        SolverCG<LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default>> solver(solver_control);

        DiagonalMatrix<LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default>> preconditioner;
        preconditioner.get_vector() = system_matrix_dev->get_matrix_diagonal_inverse()->get_vector();
        

        // Use boundary values as initial guess for CG solver
        this->solution_dev = this->boundary_values_dev;
        solver.solve(*this->system_matrix_dev, this->solution_dev, this->system_rhs_dev, preconditioner);

        // Copy solution back to host for post-processing and output
        LinearAlgebra::ReadWriteVector<double> rw_vector(this->locally_owned_dofs);
        rw_vector.import_elements(this->solution_dev, VectorOperation::insert);
        this->ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

        this->constraints.distribute(this->ghost_solution_host);
        this->ghost_solution_host.update_ghost_values();

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

        // Attach host solution vector directly (ghost values are already updated)
        data_out.add_data_vector(this->ghost_solution_host, "potential");
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
    template class MatrixFreePoissonSolverGPU<3, 1>;
    template class MatrixFreePoissonSolverGPU<3, 2>;
    template class MatrixFreePoissonSolverGPU<3, 3>;
    template class MatrixFreePoissonSolverGPU<3, 4>;
    template class MatrixFreePoissonSolverGPU<3, 5>;
}