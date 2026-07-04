#ifndef NUMERICS_MATRIX_POISSON_SOLVER_H
#define NUMERICS_MATRIX_POISSON_SOLVER_H

#include "poisson_solver_base.h"
#include "../physics/atom_system.h"

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace Numerics
{
    // Assembled-matrix Poisson solver via Trilinos. Behaviourally identical
    // to the original Poisson<dim> class; only renamed/rehomed to implement
    // PoissonSolverBase.
    template <int dim>
    class MatrixPoissonSolver : public PoissonSolverBase<dim, dealii::TrilinosWrappers::MPI::Vector>
    {
    public:
        explicit MatrixPoissonSolver(int degree, const Physics::AtomSystem<dim> &atom_system);

        void setup_system(int n_cells_per_edge,
                           double start,
                           double end,
                           const dealii::Function<dim> &boundary_condition) override;

        void assemble_system(const dealii::Function<dim> &forcing_function) override;
        int  solve(bool verbose) override;
        void output_results(const std::string &filename, bool verbose) const override;

        const dealii::TrilinosWrappers::MPI::Vector &get_solution() const override;
        const dealii::TrilinosWrappers::MPI::Vector &get_rhs() const override;

        dealii::TrilinosWrappers::SparseMatrix system_matrix;

    private:
        const Physics::AtomSystem<dim> &atom_system;
        dealii::TrilinosWrappers::MPI::Vector solution;
        dealii::TrilinosWrappers::MPI::Vector system_rhs;
        dealii::TrilinosWrappers::MPI::Vector system_rhs_before;
        dealii::TrilinosWrappers::PreconditionJacobi jacobi_preconditioner;
    };
}

#endif