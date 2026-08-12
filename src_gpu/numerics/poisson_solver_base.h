#ifndef NUMERICS_POISSON_SOLVER_BASE_H
#define NUMERICS_POISSON_SOLVER_BASE_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <string>

namespace Numerics
{
  // This is a base class for Poisson solvers that can be implemented on either CPU or GPU.
  // For CPU, the vector type uses dealii::MemorySpace::Host
  // For GPU, the vector type uses dealii::MemorySpace::Cuda (dealii::MemorySpace::Default)
  using HostVector = dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>;

  template <int dim>
  class PoissonSolverBase
  {
  public:
    explicit PoissonSolverBase(int degree, MPI_Comm comm = MPI_COMM_WORLD)
      : mpi_communicator(comm)
      , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      , triangulation(mpi_communicator)
      , dof_handler(triangulation)
      , fe(degree)
    {}

    virtual ~PoissonSolverBase() = default;

    // Prevent accidental copying
    PoissonSolverBase(const PoissonSolverBase &) = delete;
    PoissonSolverBase &operator=(const PoissonSolverBase &) = delete;

    // Pure virtual solver interface
    virtual void setup_system(int n_cells_per_edge,
                              double start,
                              double end,
                              const dealii::Function<dim> &boundary_condition) = 0;

    virtual void assemble_system(const dealii::Function<dim> &forcing_function) = 0;
    virtual int  solve(bool verbose) = 0;
    virtual void output_results(const std::string &filename, bool verbose) const = 0;

    // Unified Host Accessors (Supported by both CPU and GPU derived classes)
    virtual const HostVector &get_solution_host() const = 0;

    // Convenience driver
    void run(int n_cells_per_edge, double start, double end,
             const dealii::Function<dim> &bc, const dealii::Function<dim> &rhs,
             bool verbose = false);

    const dealii::parallel::distributed::Triangulation<dim> &get_triangulation() const
    {
        return triangulation;
    }

    const dealii::DoFHandler<dim> &get_dof_handler() const
    {
        return dof_handler;
    }

    const dealii::FE_Q<dim> &get_fe() const
    {
        return fe;
    }

    unsigned int get_n_dofs() const
    {
        return dof_handler.n_dofs();
    }

    unsigned int get_n_cells() const
    {
        return triangulation.n_global_active_cells();
    }

  protected:
    MPI_Comm                                           mpi_communicator;
    dealii::ConditionalOStream                         pcout;
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    dealii::DoFHandler<dim>           dof_handler;
    dealii::FE_Q<dim>                 fe;
    dealii::AffineConstraints<double> constraints;

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;
  };
}

#endif // NUMERICS_POISSON_SOLVER_BASE_H