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

using namespace dealii;

#define r_c 0.5 // Less than half the distance between the two nuclei
#define r_c_squared r_c * r_c
#define r_c_8 r_c_squared * r_c_squared * r_c_squared * r_c_squared
#define PI 3.14159265358979323846


template <int dim>
class SmearedCharge : public Function<dim> // Inherit from Function class
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
class CoullombPotential : public Function<dim> // Inherit from Function class
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
                return charge / 1e-12;

            return charge / r;
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

    private:
        void setup_system(int refinement_level);
        void assemble_system(SmearedCharge<dim> &forcing_function, CoullombPotential<dim> &boundary_condition);
        void solve(bool verbose);
        void output_results(bool verbose) const;

        // Mesh, finite element (basis functions), and dof handler (dof to mesh mapping)
        Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        // Containers
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
        Vector<double> system_rhs_before;
};

template <int dim>
Poisson<dim>::Poisson(int degree) : fe(degree), dof_handler(triangulation)
{
    
}

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
void Poisson<dim>::setup_system(int n_cells_per_edge)
{
    // Make grid and distribute dofs

    GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, -1.0, 1.0); // Creates a grid of n_cells_per_edge^dim cells in the unit cube [-1, 1]^dim

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
    QGauss<dim> quadrature_formula(fe.degree + 1); // Quadrature
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points); 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    (void) n_q_points;

    // Local cell matrix and rhs vector
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    // Local array mapping local DoF index to global matrix index
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell); 

    // a_ij =  summation ( grad phi_i * grad phi_j * jacobian * quadrature weight )
    // b_i =  summation ( f(x) * phi_i * jacobian * quadrature weight )
    // phi_i and phi_j are the basis functions
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell); // When different cells have different geometry

        // Pointer to quadrature points for this cell
        const std::vector<Point<dim>> &quadrature_points = fe_values.get_quadrature_points();

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            const double f_q = forcing_function.value(quadrature_points[q_index]);

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index); 
                }
                cell_rhs(i) += f_q * fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
            }
        }

        // Fills local_dof_indices with the true global matrix indices for this cell
        // Review
        cell->get_dof_indices(local_dof_indices);

        // Add local contributions to global matrix and rhs vector
        for (const unsigned int i : fe_values.dof_indices())
        {
            for (const unsigned int j : fe_values.dof_indices())
            {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    // Important!
    system_rhs_before = system_rhs;

    // USEFULL
    std::map<types::global_dof_index, double> boundary_values;
    
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template <int dim>
void Poisson<dim>::solve(bool verbose)
{
    SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);

    // Initially no preconditioner
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

    if(verbose)
    {
        std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }
}

template <int dim>
void Poisson<dim>::output_results(bool verbose) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "potential");
    data_out.build_patches();
    std::ofstream output("poisson-solution.vtu");
    data_out.write_vtu(output);
    
    if (verbose)
    {
        std::cout << "  Results written to poisson-solution.vtu" << std::endl;
    }
}

template <int dim>
void Poisson<dim>::run(int refinement_level, bool verbose)
{
    if (verbose)
    {
        std::cout << "Setting up system with refinement level " << refinement_level << "..." << std::endl;
    }

    setup_system(refinement_level); // Now dynamically controlled!

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

double electrostatic_energy(const Vector<double> &sol, const Vector<double> &rhs)
{
    double I =  sol * rhs; 

    return -0.5 * I;
}

int main()
{
    int degree = 2;

    for(int n_cells_per_edge = 2; n_cells_per_edge <= 20; n_cells_per_edge += 2)
    {
        // std::cout << "Running Poisson solver with refinement level " << refinement_level << "..." << std::endl;
        Poisson<3> poisson_problem(degree);
        poisson_problem.run(n_cells_per_edge);

        const Vector<double> &solution = poisson_problem.get_solution();
        const Vector<double> &rhs = poisson_problem.get_rhs();

        //Total charge
        cout<< "Total Charge: " << rhs.l1_norm() << std::endl;

        // Debug
        cout<< "Solution vector size: " << solution.size() << std::endl;
        // cout<< "RHS vector size: " << rhs.size() << std::endl;

        double energy = electrostatic_energy(solution, rhs);
        std::cout << "Electrostatic energy at n_cells_per_edge " << n_cells_per_edge << ": " << energy << std::endl;
    }
}

