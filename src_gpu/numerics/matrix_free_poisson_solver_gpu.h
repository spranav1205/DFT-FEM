#ifndef NUMERICS_MATRIX_FREE_POISSON_SOLVER_GPU_H
#define NUMERICS_MATRIX_FREE_POISSON_SOLVER_GPU_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "laplace_operator_gpu.h"
#include "../physics/atom_system.h"
#include "poisson_solver_base.h"

#include <memory>

namespace Numerics
{
  using namespace dealii;

  template <int dim, int fe_degree>
  class MatrixFreePoissonSolverGPU
    : public PoissonSolverBase<dim>
  {
  public:
    using SystemMatrixType = LaplaceOperatorGPU<dim, fe_degree, double>;
    using VectorTypeDevice = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

    MatrixFreePoissonSolverGPU(int degree,
                               const Physics::AtomSystem<dim> &atom_system);

    void setup_system(int n_cells_per_edge,
                      double start,
                      double end,
                      const Function<dim> &boundary_condition) override;

    // Overrides pure virtual interface in PoissonSolverBase<dim>
    void assemble_system(const dealii::Function<dim> &forcing_function) override;

    int solve(bool verbose = true) override;

    void output_results(const std::string &filename,
                         bool verbose = true) const override;

    // Overrides Host accessor in PoissonSolverBase<dim>
    const HostVector &get_solution_host() const override;

    // Concrete accessor for host RHS
    const HostVector &get_rhs() const;

  public:
    const Physics::AtomSystem<dim> &atom_system;

    // GPU Operator
    std::unique_ptr<SystemMatrixType> system_matrix_dev;

    // Device (GPU) Vectors
    VectorTypeDevice boundary_values_dev;
    VectorTypeDevice solution_dev;
    VectorTypeDevice system_rhs_dev;

    // Host (CPU) Vectors
    HostVector system_rhs_host;
    HostVector ghost_solution_host;
  };
}

#endif // NUMERICS_MATRIX_FREE_POISSON_SOLVER_GPU_H