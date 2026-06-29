#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h> // Added for parallel sparsity distribution
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>

// MPI
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
// Parallel: Added solver wrapper header for Trilinos
#include <deal.II/lac/trilinos_solver.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>
#include <iomanip>  // Required for std::setprecision
#include <chrono>   // Required for high-precision timing
#include <mpi.h>

using namespace dealii;

#define r_c 0.7 // Less than half the distance between the two nuclei
#define r_c_squared (r_c * r_c)
#define r_c_8 (r_c_squared * r_c_squared * r_c_squared * r_c_squared)
#define PI 3.14159265358979323846

template <int dim>
class SmearedCharge : public Function<dim> 
{
    public:
        SmearedCharge() : Function<dim>() {}
        double charge = 1.0;

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void) component; // Unused parameter

            double r_squared = 0;
            for (int i = 0; i < dim; ++i)
            {
                r_squared += p[i] * p[i];
            }
            
            double r = std::sqrt(r_squared);

            if(r <= r_c)
            {
                double n = -21.0 * (r - r_c) * (r - r_c) * (r - r_c) * (6.0 * r_squared + 3.0 * r * r_c + r_c_squared);
                double d = 5.0 * PI * r_c_8;
                return charge*n / d;
            }
            else
            {
                return 0.0; 
            }
        }
};

template <int dim>
class CoullombPotential : public Function<dim> 
{
    public:
        CoullombPotential() : Function<dim>() {}
        double charge = 1.0;

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void) component; // Unused parameter
            double r_squared = 0;
            for (int i = 0; i < dim; ++i)
            {
                r_squared += p[i] * p[i];
            }

            double r = std::sqrt(r_squared);

            if (r < 1e-12)
                return charge / 1e-12; // TODO

            return charge / (4.0 * M_PI * r);
        }
};

template <int dim>
class Poisson
{
    public:
        Poisson(int degree);
        void run(int refinement_level, bool verbose = false);

        // The solution and right-hand-side vectors are distributed Trilinos vectors.
        const TrilinosWrappers::MPI::Vector &get_solution() const;
        const TrilinosWrappers::MPI::Vector &get_rhs() const;
        
        void setup_system(int n_cells_per_edge, double start = -1.0, double end = 1.0, const CoullombPotential<dim> &boundary_condition = CoullombPotential<dim>());
        void assemble_system(SmearedCharge<dim> &forcing_function);
        int solve(bool verbose);
        void output_results(const std::string &filename, bool verbose) const;

        // The mesh is partitioned among MPI ranks; each rank owns only a subset of the cells.
        parallel::distributed::Triangulation<dim> triangulation; // Parallel
        MPI_Comm mpi_communicator; // Parallel communicator
        ConditionalOStream pcout; // Rank-aware output stream: only rank 0 writes to the terminal.

        DoFHandler<dim> dof_handler;
        FE_Q<dim> fe;
        AffineConstraints<double> constraints; // Hanging-node and boundary constraints shared across ranks.

        // New Containers
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        TrilinosWrappers::SparseMatrix system_matrix;
        TrilinosWrappers::MPI::Vector  solution;
        TrilinosWrappers::MPI::Vector  system_rhs;
        TrilinosWrappers::MPI::Vector  system_rhs_before;
        TrilinosWrappers::PreconditionJacobi jacobi_preconditioner; 
};

template <int dim>
Poisson<dim>::Poisson(int degree) : 
    triangulation(MPI_COMM_WORLD, Triangulation<dim>::limit_level_difference_at_vertices),
    mpi_communicator(MPI_COMM_WORLD), 
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    dof_handler(triangulation),
    fe(degree)
{ }


// Parallel: Return TrilinosWrappers::MPI::Vector instead of deal.II Vector
template <int dim>
const TrilinosWrappers::MPI::Vector& Poisson<dim>::get_solution() const
{
    return solution;
}

// Parallel: Return TrilinosWrappers::MPI::Vector instead of deal.II Vector
template <int dim>
const TrilinosWrappers::MPI::Vector& Poisson<dim>::get_rhs() const
{
    return system_rhs_before;
}

template <int dim>
void Poisson<dim>::setup_system(int n_cells_per_edge, double start, double end, const CoullombPotential<dim> &boundary_condition)
{
    int marked = 0;
    GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);
    for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
        // Each rank checks only its owned cells and marks cells for refinement locally.
        marked = 0;
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if(cell->is_locally_owned())
            {
                if (cell->center().norm() < 2.0)
                {
                    cell->set_refine_flag();
                    marked++;
                }
            }
        }

        // Sum the locally marked cells from all ranks into one global count.
        int total_marked = Utilities::MPI::sum(marked, mpi_communicator);
        pcout << "Marked " << total_marked << " cells\n";
        triangulation.execute_coarsening_and_refinement();
    }

    pcout<< "Number of active cells: " << triangulation.n_global_active_cells() << std::endl; 
    dof_handler.distribute_dofs(fe); // Assigns dofs to the mesh vertices, edges, faces, and cells

    // Identify the DoFs owned by this rank and the neighbor DoFs needed as ghosts.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    constraints.clear();
    // Constraints must be reinitialized with the locally relevant DoF set because they can refer to ghosts.
    // Reinitializing -> Resizing
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
    constraints.close();
    // TODO: SUM 
    pcout<< "Number of constraints: " << constraints.n_constraints() << std::endl;
    
    // Sparsity pattern -> list of possibly non-zero entries in the matrix
    // Process 1 discovers that row 4 must contain columns 1,2,3,4 in the global matrix.
    //
    // distribute_sparsity_pattern() sends this row structure
    // to Process 2 (the owner of row 4).
    //
    // Process 2 allocates storage for A(4,1), A(4,2), A(4,3), A(4,4) before assembly begins.

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

    // Reinit the sparse matrix using the parallelly synchronized sparsity pattern
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

    // The right-hand side is distributed the same way as the matrix rows.
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs_before.reinit(locally_owned_dofs, mpi_communicator);
    // The solved vector is owned-only so constrained values can be written back into it.
    solution.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void Poisson<dim>::assemble_system(SmearedCharge<dim> &forcing_function)
{
    QGauss<dim> matrix_quad(fe.degree + 1); // Quadrature
    QIterated<dim> rhs_quad(QGauss<1>(6), 2); // 6 points per interval, 2 intervals per edge -> 8 points per edge -> 512 points in total for 3D
    FEValues<dim> fe_matrix(fe, matrix_quad, update_values | update_gradients | update_JxW_values); 
    FEValues<dim> fe_rhs(fe, rhs_quad, update_values | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // Local cell matrix and rhs vector
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell); 

    // a_ij =  summation ( grad phi_i * grad phi_j * jacobian * quadrature weight )
    // b_i =  summation ( f(x) * phi_i * jacobian * quadrature weight )
    // phi_i and phi_j are the basis functions

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        // Every rank iterates over the full active-cell range, but only the owner assembles each cell once.
        if (cell->is_locally_owned())
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_matrix.reinit(cell);
            fe_rhs.reinit(cell);

            // Pointer to quadrature points for this cell
            const std::vector<Point<dim>> &quadrature_points_r = fe_rhs.get_quadrature_points();

            for (const unsigned int q_index : fe_matrix.quadrature_point_indices())
            {
                for (const unsigned int i : fe_matrix.dof_indices())
                {
                    for (const unsigned int j : fe_matrix.dof_indices())
                    {
                        cell_matrix(i, j) += fe_matrix.shape_grad(i, q_index) * fe_matrix.shape_grad(j, q_index) * fe_matrix.JxW(q_index); 
                    }
                }
            }

            for(const unsigned int q_index : fe_rhs.quadrature_point_indices())
            {
                const double f_q = forcing_function.value(quadrature_points_r[q_index]);

                for (const unsigned int i : fe_rhs.dof_indices())
                {
                    cell_rhs(i) += f_q * fe_rhs.shape_value(i, q_index) * fe_rhs.JxW(q_index);
                }
            }

            // Fills local_dof_indices with the true global matrix indices for this cell
            cell->get_dof_indices(local_dof_indices);

            // Apply hanging-node and boundary constraints while scattering local element contributions into the
            // distributed Trilinos matrix and right-hand side.
            // If the row belongs to some other rank, the contribution is buffered and sent to the owning rank during compress().
            // Say process 0 will send the contribution to process 1 for row 4, and process 1 will add it to its local copy of row 4.
            constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        }
    }

    // Finalize the distributed assembly by summing buffered contributions from all ranks.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    system_rhs_before = system_rhs;

    // Jacobi preconditioner
    TrilinosWrappers::PreconditionJacobi preconditioner;
    TrilinosWrappers::PreconditionJacobi::AdditionalData jacobi_data;
    jacobi_data.omega = 1.0;   // damping parameter
    preconditioner.initialize(system_matrix, jacobi_data);
}

template <int dim>
int Poisson<dim>::solve(bool verbose)
{
    // Use relative residual scaling based on the L2 norm of the RHS vector to match parallel sizing
    SolverControl solver_control(1000, 1e-10);
    // Trilinos CG solves the distributed linear system using the MPI-aware matrix and vectors.
    TrilinosWrappers::SolverCG solver(solver_control);

    // solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);
    // Distribute constrained values back into the ghosted solution vector after the solve.
    constraints.distribute(solution);

    if(verbose)
    {
        // Parallel: Output via pcout
        pcout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }

    return solver_control.last_step();
}

template <int dim>
void Poisson<dim>::output_results(const std::string &filename, bool verbose) const
{
    // Each rank writes its own .vtu partition file, while rank 0 also writes the .pvtu master record.
    std::string partition_filename = filename;
    const unsigned int step = Utilities::MPI::this_mpi_process(mpi_communicator);
    const std::string step_string = "." + Utilities::int_to_string(step, 4);
    const std::string::size_type ext_pos = partition_filename.rfind(".vtu");
    if (ext_pos != std::string::npos)
        partition_filename.insert(ext_pos, step_string);
    else
        partition_filename += step_string + ".vtu";

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    TrilinosWrappers::MPI::Vector output_solution;
    output_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    output_solution = solution;
    output_solution.update_ghost_values();
    data_out.add_data_vector(output_solution, "potential");
    data_out.build_patches(fe.degree); // TODO Fix export
    
    // Save to the dedicated folder path provided in the filename string
    std::ofstream output(partition_filename);
    data_out.write_vtu(output);
    
    if (verbose)
    {
        // Only rank 0 reports the shared filename to keep logging readable.
        pcout << "  Results written to partitioned processors matching: " << filename << std::endl;
    }

    // Rank 0 writes the metadata file that tells ParaView how to assemble the per-rank pieces.
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
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

template <int dim>
void Poisson<dim>::run(int n_cells_per_edge, bool verbose)
{
    if (verbose)
    {
        pcout << "Setting up system with " << n_cells_per_edge << " cells per edge..." << std::endl;
    }

    setup_system(n_cells_per_edge, -1.0, 1.0, CoullombPotential<dim>());

    SmearedCharge<dim> forcing_term;
    CoullombPotential<dim> boundary_term;

    forcing_term.charge = 3.0; 
    boundary_term.charge = 3.0;


    if (verbose)
    {
        pcout << "Assembling matrix and vectors..." << std::endl;
    }

    assemble_system(forcing_term);

    if(verbose)
    {
        pcout << "Solving linear system via CG..." << std::endl;
    }

    solve(verbose);

    if(verbose)
    {
        pcout << "Generating output files..." << std::endl;
    }

    output_results("temp.vtu", verbose);
    
}


double charge3D(SmearedCharge<3> &forcing_function, FE_Q<3>& fe, DoFHandler<3>& dof_handler)
{
    double total_charge = 0.0;
    // Keep a local scalar accumulator on each rank before the MPI reduction.
    double local_charge = 0.0;

    QIterated<3> quad(QGauss<1>(6), 2); // 6 points per interval, 2 intervals per edge 
    FEValues<3> fe_values(fe, quad, update_values | update_JxW_values | update_quadrature_points);

    // update_values: value if basis function at quadrature point
    // update_JxW_values: jacobian * quadrature weight at quadrature point
    // update_quadrature_points: coordinates of quadrature points

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        // Only the owning rank integrates a given cell to avoid duplicated contributions.
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            const std::vector<Point<3>> &quadrature_points = fe_values.get_quadrature_points();

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                local_charge += forcing_function.value(quadrature_points[q_index]) * fe_values.JxW(q_index);
            }
        }
    }

    // Reduce the local charge contributions from every rank to one global scalar integral.
    total_charge = Utilities::MPI::sum(local_charge, MPI_COMM_WORLD);
    return total_charge;
}

// Parallel: Changed vector signature to utilize parallel Trilinos type
double electrostatic_energy(SmearedCharge<3> &forcing_function, DoFHandler<3>& dof_handler, FE_Q<3>& fe, const TrilinosWrappers::MPI::Vector& solution)
{
    double energy = 0.0;
    // Keep a local scalar accumulator on each rank before the MPI reduction.
    double local_energy = 0.0;

    QIterated<3> quad(QGauss<1>(6), 2); // 6 points per interval, 2 intervals per edge -> 20 points per edge -> 8000 points in total for 3D
    FEValues<3> fe_values(fe, quad, update_values | update_JxW_values | update_quadrature_points);

    // Build a ghosted copy of the solved vector so each rank can read neighbor-owned DoFs during interpolation.
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    TrilinosWrappers::MPI::Vector solution_with_ghosts;
    solution_with_ghosts.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_with_ghosts = solution;
    solution_with_ghosts.update_ghost_values();

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        // Only the owning rank evaluates the energy contribution for each cell.
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);

            // Get quadrature points and solution values at quadrature points
            const std::vector<Point<3>> &quadrature_points = fe_values.get_quadrature_points();
            std::vector<double> solution_values(fe_values.n_quadrature_points);
            fe_values.get_function_values(solution_with_ghosts, solution_values);

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                double phi_q = solution_values[q_index];
                double rho_q = forcing_function.value(quadrature_points[q_index]);
                local_energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index); // Jacobian * quadrature weight
            }
        }
    }

    // Reduce the local energy contributions from every rank to one global scalar integral.
    energy = Utilities::MPI::sum(local_energy, MPI_COMM_WORLD);
    return energy;
}

int main(int argc, char *argv[]) // CLI: ./poisson_102_parallel n_cells_per_edge
{
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        Poisson<3> poisson_problem(3);  
        SmearedCharge<3> forcing_term;
        CoullombPotential<3> boundary_term; 
        
        forcing_term.charge = 12.0;
        boundary_term.charge = 12.0;

        int n_cells_per_edge = 12;
        if (argc > 1)
            n_cells_per_edge = std::atoi(argv[1]);

        poisson_problem.setup_system(n_cells_per_edge, -10.0, 10.0, boundary_term);
        poisson_problem.assemble_system(forcing_term);

        // double total_charge = charge3D(forcing_term, poisson_problem.fe, poisson_problem.dof_handler);
        // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) std::cout << "Total charge in the system: " << total_charge << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        int num_iterations = poisson_problem.solve(false);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_time = end_time - start_time;
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
            std::cout << "Solver completed in " << elapsed_time.count() << " seconds." << std::endl;
            std::cout << "Number of CG iterations: " << num_iterations << std::endl;
        }
        
        poisson_problem.output_results("./parallel_output/temp.vtu", true);

        double energy = electrostatic_energy(forcing_term, poisson_problem.dof_handler, poisson_problem.fe, poisson_problem.get_solution());
        
        // Parallel: Ensure only rank 0 displays final calculated data to clear clutter
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
            std::cout << "Electrostatic energy of the system: " << std::fixed << std::setprecision(8) << energy << std::endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}