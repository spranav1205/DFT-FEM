#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace dealii;

template <int dim>
class MinimalStiffnessMatrix
{
public:
  MinimalStiffnessMatrix();
  void run();

private:
  void setup_system();
  void assemble_matrix();
  void print_matrix();

  Triangulation<dim>        triangulation;
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
};

template <int dim>
MinimalStiffnessMatrix<dim>::MinimalStiffnessMatrix()
  : fe(4), dof_handler(triangulation)
{}

template <int dim>
void MinimalStiffnessMatrix<dim>::setup_system()
{
  // 1. Initial grid on [-2, 2]^dim
  GridGenerator::subdivided_hyper_cube(triangulation, 5, -2.0, 2.0);

  // 2. Define 10 refinement points (coplanar / in same plane)
  std::vector<Point<dim>> refinement_points;
  refinement_points.reserve(10);

  // Example: 10 points arranged along a circle of radius 1.2 in the xy-plane
  const double radius = 1.2;
  for (unsigned int k = 0; k < 10; ++k)
    {
      const double angle = 2.0 * M_PI * k / 10.0;
      Point<dim> p;
      p[0] = radius * std::cos(angle);
      p[1] = radius * std::sin(angle);
      if (dim == 3)
        p[2] = 0.0; // Keep all points in the z = 0 plane for 3D

      refinement_points.push_back(p);
    }

  const double refinement_radius = 0.7; // Sphere/circle radius around each point

  // 3. Perform local adaptive refinement around all 10 points
  for (unsigned int cycle = 0; cycle < 1; ++cycle)
    {
      unsigned int marked_cells = 0;
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          bool mark_for_refinement = false;

          for (const auto &ref_pt : refinement_points)
            {
              if (cell->center().distance(ref_pt) < refinement_radius)
                {
                  mark_for_refinement = true;
                  break; // Cell is close to at least one point, no need to check others
                }
            }

          if (mark_for_refinement)
            {
              cell->set_refine_flag();
              marked_cells++;
            }
        }

      std::cout << "Cycle " << cycle << ": Marked " << marked_cells 
                << " cells for refinement." << std::endl;
      triangulation.execute_coarsening_and_refinement();
    }

  // 4. Distribute degrees of freedom
  dof_handler.distribute_dofs(fe);

  // 5. Build constraints (Hanging nodes + Zero Dirichlet Boundary)
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  // 6. Initialize Matrix Structure
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void MinimalStiffnessMatrix<dim>::assemble_matrix()
{
  QGauss<dim>   quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature, update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            {
              for (const unsigned int j : fe_values.dof_indices())
                {
                  cell_matrix(i, j) += fe_values.shape_grad(i, q_index) *
                                       fe_values.shape_grad(j, q_index) *
                                       fe_values.JxW(q_index);
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);

      // Distribute local matrix directly without RHS vector
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
    }
}

template <int dim>
void MinimalStiffnessMatrix<dim>::print_matrix()
{
  const unsigned int n_dofs = dof_handler.n_dofs();

  std::cout << "\nSaving " << n_dofs << "x" << n_dofs 
            << " stiffness matrix to 'stiffness_matrix.txt'..." << std::endl;

  std::ofstream out("stiffness_matrix.txt");
  system_matrix.print_formatted(out, 6, false, 0, "0.0");
  out.close();
  std::cout << "Done!" << std::endl;
}

template <int dim>
void MinimalStiffnessMatrix<dim>::run()
{
  setup_system();
  assemble_matrix();
  print_matrix();
}

int main()
{
  MinimalStiffnessMatrix<2> problem;
  problem.run();
  return 0;
}