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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <iomanip>  // Required for std::setprecision
#include <chrono>   // Required for high-precision timing

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

        const Vector<double> &get_solution() const;
        const Vector<double> &get_rhs() const;
        
        void setup_system(int n_cells_per_edge, double start = -1.0, double end = 1.0);
        void assemble_system(SmearedCharge<dim> &forcing_function, CoullombPotential<dim> &boundary_condition);
        int solve(bool verbose);
        void output_results(const std::string &filename, bool verbose) const;

        // Mesh, finite element (basis functions), and dof handler (dof to mesh mapping)
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        PreconditionJacobi<SparseMatrix<double>> jacobi_preconditioner;
        FE_Q<dim> fe;

        // Containers
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
        Vector<double> system_rhs_before;
};

template <int dim>
Poisson<dim>::Poisson(int degree) : fe(degree), dof_handler(triangulation)
{ }

// Returns pointer to solution vector
template <int dim>
const Vector<double>& Poisson<dim>::get_solution() const
{
    return solution;
}

template <int dim>
const Vector<double>& Poisson<dim>::get_rhs() const
{
    return system_rhs_before;
}

template <int dim>
void Poisson<dim>::setup_system(int n_cells_per_edge, double start, double end)
{
    // Make grid and distribute dofs
    GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);
    cout<< "Number of active cells: " << triangulation.n_active_cells() << std::endl; 

    dof_handler.distribute_dofs(fe); // Assigns dofs to the mesh vertices, edges, faces, and cells
    
    // Make big matrix -> see where the non-zero entries are -> make sparse matrix with non-zero entries only
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    system_rhs_before.reinit(dof_handler.n_dofs());
}

template <int dim>
void Poisson<dim>::assemble_system(SmearedCharge<dim> &forcing_function, CoullombPotential<dim> &boundary_condition)
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
        // Review
        cell->get_dof_indices(local_dof_indices);

        // Add local contributions to global matrix and rhs vector
        for (const unsigned int i : fe_matrix.dof_indices())
        {
            for (const unsigned int j : fe_matrix.dof_indices())
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    // Why?
    system_rhs_before = system_rhs;

    // USEFULL
    std::map<types::global_dof_index, double> boundary_values;
    
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

    // Preconditioner setup
    jacobi_preconditioner.initialize(system_matrix);
}

template <int dim>
int Poisson<dim>::solve(bool verbose)
{
    SolverControl solver_control(1000, 1e-10); // Max 1000 iterations, tolerance 1e-10 i.e. ||Ax-b|| < 1e-10
    SolverCG<Vector<double>> solver(solver_control);

    // solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);

    if(verbose)
    {
        std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }

    return solver_control.last_step();
}

template <int dim>
void Poisson<dim>::output_results(const std::string &filename, bool verbose) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "potential");
    data_out.build_patches();
    
    // Save to the dedicated folder path provided in the filename string
    std::ofstream output(filename);
    data_out.write_vtu(output);
    
    if (verbose)
    {
        std::cout << "  Results written to " << filename << std::endl;
    }
}

template <int dim>
void Poisson<dim>::run(int n_cells_per_edge, bool verbose)
{
    if (verbose)
    {
        std::cout << "Setting up system with " << n_cells_per_edge << " cells per edge..." << std::endl;
    }

    setup_system(n_cells_per_edge); // Now dynamically controlled!

    SmearedCharge<dim> forcing_term;
    CoullombPotential<dim> boundary_term;

    forcing_term.charge = 3.0; 
    boundary_term.charge = 3.0;


    if (verbose)
    {
        std::cout << "Assembling matrix and vectors..." << std::endl;
    }

    assemble_system(forcing_term, boundary_term);

    if(verbose)
    {
        std::cout << "Solving linear system via CG..." << std::endl;
    }

    solve(verbose);

    if(verbose)
    {
        std::cout << "Generating output files..." << std::endl;
    }

    output_results(verbose);
    
}


double charge3D(SmearedCharge<3> &forcing_function, FE_Q<3>& fe, DoFHandler<3>& dof_handler)
{
    double total_charge = 0.0;

    QIterated<3> quad(QGauss<1>(6), 2); // 6 points per interval, 2 intervals per edge 
    FEValues<3> fe_values(fe, quad, update_values | update_JxW_values | update_quadrature_points);

    // update_values: value if basis function at quadrature point
    // update_JxW_values: jacobian * quadrature weight at quadrature point
    // update_quadrature_points: coordinates of quadrature points

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        const std::vector<Point<3>> &quadrature_points = fe_values.get_quadrature_points();

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            total_charge += forcing_function.value(quadrature_points[q_index]) * fe_values.JxW(q_index);
        }
    }

    return total_charge;
}

double electrostatic_energy(SmearedCharge<3> &forcing_function, DoFHandler<3>& dof_handler, FE_Q<3>& fe, const Vector<double>& solution)
{
    double energy = 0.0;

    QIterated<3> quad(QGauss<1>(6), 2); // 6 points per interval, 2 intervals per edge -> 20 points per edge -> 8000 points in total for 3D
    FEValues<3> fe_values(fe, quad, update_values | update_JxW_values | update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        // Get quadrature points and solution values at quadrature points
        const std::vector<Point<3>> &quadrature_points = fe_values.get_quadrature_points();
        std::vector<double> solution_values(fe_values.n_quadrature_points);
        fe_values.get_function_values(solution, solution_values);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            double phi_q = solution_values[q_index];
            double rho_q = forcing_function.value(quadrature_points[q_index]);
            energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index); // Jacobian * quadrature weight
        }
    }

    return energy;
}

// int main()
// {
//     Poisson<3> poisson_problem(3);  
//     SmearedCharge<3> forcing_term;
//     CoullombPotential<3> boundary_term;

//     forcing_term.charge = 13.0;
//     boundary_term.charge = 13.0;

//     poisson_problem.setup_system(16, -10.0, 10.0);
//     poisson_problem.assemble_system(forcing_term, boundary_term);

//     double total_charge = charge3D(forcing_term, poisson_problem.fe, poisson_problem.dof_handler);
//     std::cout << "Total charge in the system: " << total_charge << std::endl;

//     poisson_problem.solve(true);
//     poisson_problem.output_results(true);

//     double energy = electrostatic_energy(forcing_term, poisson_problem.dof_handler, poisson_problem.fe, poisson_problem.get_solution());
//     std::cout << "Electrostatic energy of the system: " << energy << std::endl;
// }

int main()
{
    SmearedCharge<3> forcing_term;
    CoullombPotential<3> boundary_term;

    forcing_term.charge = 12.0;
    boundary_term.charge = 12.0;

    std::ofstream file("output/energy_convergence.csv");
    
    file << "cells_per_edge,total_cells,total_charge,energy,steps,solver_time_sec\n";

    for (unsigned int cells_per_edge = 16;
         cells_per_edge <= 48;
         cells_per_edge += 8)
    {
        std::cout << "\n=====================================\n";
        std::cout << "Cells per edge = " << cells_per_edge << std::endl;

        Poisson<3> poisson_problem(3);

        poisson_problem.setup_system(cells_per_edge, -5.0, 5.0);
        poisson_problem.assemble_system(forcing_term, boundary_term);

        const double total_charge = charge3D(
            forcing_term,
            poisson_problem.fe,
            poisson_problem.dof_handler);

        // Time measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int steps = poisson_problem.solve(true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    
        // Calculation
        const double energy = electrostatic_energy(
            forcing_term,
            poisson_problem.dof_handler,
            poisson_problem.fe,
            poisson_problem.get_solution());

        const unsigned int total_cells = poisson_problem.triangulation.n_active_cells();

        file << cells_per_edge << ","
             << total_cells << ","
             << total_charge << ","
             << std::fixed << std::setprecision(8) << energy << ","
             << steps << ","
             << std::defaultfloat << elapsed_seconds.count() << "\n";

        std::string vtu_filename = "output/poisson-solution-" + std::to_string(cells_per_edge) + "cells.vtu";
        poisson_problem.output_results(vtu_filename, true);

        std::cout << "Cells          : " << total_cells << std::endl;
        std::cout << "Total charge   : " << total_charge << std::endl;
        std::cout << "Energy         : " << std::fixed << std::setprecision(8) << energy << std::endl;
        std::cout << "Solver Time    : " << std::defaultfloat << elapsed_seconds.count() << " sec" << std::endl;
    }

    file.close();
    std::cout << "\nAll results written successfully to the 'output/' directory.\n";

    return 0;
}