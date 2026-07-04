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

#include <string>

namespace Numerics
{
    // Abstract interface shared by every Poisson solver backend
    // (assembled-matrix, matrix-free, ...). VectorType is the backend's
    // native solution/RHS vector type (e.g. TrilinosWrappers::MPI::Vector
    // for an assembled solver, LinearAlgebra::distributed::Vector<double>
    // for a matrix-free one).
    template <int dim, typename VectorType>
    class PoissonSolverBase
    {
    public:
        explicit PoissonSolverBase(int degree)
            :mpi_communicator(MPI_COMM_WORLD)
            , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            , triangulation(mpi_communicator)
            , dof_handler(triangulation)
            , fe(degree)
            {}
        virtual ~PoissonSolverBase() = default;

        virtual void setup_system(int n_cells_per_edge,
                                   double start,
                                   double end,
                                   const dealii::Function<dim> &boundary_condition) = 0;

        virtual void assemble_system(const dealii::Function<dim> &forcing_function) = 0;
        virtual int  solve(bool verbose) = 0;
        virtual void output_results(const std::string &filename, bool verbose) const = 0;

        virtual const VectorType &get_solution() const = 0;
        virtual const VectorType &get_rhs() const = 0;

        // Convenience driver, same shape as the original Poisson::run().
        void run(int n_cells_per_edge, bool verbose = false);

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

#endif