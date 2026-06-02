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

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class Poisson
{
    public:
        Poisson(int degree);
        void run();

    private:
        void setup_system(int refinement_level);
        void assemble_system();
        void solve();
        void output_results() const;

        // Mesh, finite element (basis functions), and dof handler (dof to mesh mapping)
        Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

        // Containers
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
};

template <int dim>
Poisson<dim>::Poisson(int degree) : fe(degree), dof_handler(triangulation)
{
    
}

template <int dim>
void Poisson<dim>::setup_system(int refinement_level)
{
    // Make grid and distribute dofs

    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refinement_level); // n_cells = 2^(dim*refinement_level)
    dof_handler.distribute_dofs(fe); // Assigns dofs to the mesh vertices, edges, faces, and cells
    
    // Make big matrix -> see where the non-zero entries are -> make sparse matrix with non-zero entries only
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    SparsityTools::distribute_dofs(dof_handler, dsp);
    system_matrix.reinit(dsp);

}

template <int dim>
void Poisson<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(fe.degree + 1); // Quadrature
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values); 

    // https://dealii.org/developer/doxygen/deal.II/step_3.html

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    // Local cell matrix and rhs vector
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    // Local only, badly named datastructure -_-
    vector<types::global_dof_index> local_dof_indices(dofs_per_cell); 