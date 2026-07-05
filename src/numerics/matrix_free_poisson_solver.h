#ifndef NUMERICS_MATRIX_FREE_POISSON_SOLVER_H
#define NUMERICS_MATRIX_FREE_POISSON_SOLVER_H

#include "poisson_solver_base.h"
#include "../physics/atom_system.h"

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

namespace Numerics
{
    // Matrix-free Laplace operator: applies -Delta u cell-by-cell via
    // FEEvaluation, and supplies the diagonal for a Jacobi preconditioner.
    template <int dim, int fe_degree, typename number = double>
    class LaplaceOperator
        : public dealii::MatrixFreeOperators::
              Base<dim, dealii::LinearAlgebra::distributed::Vector<number>>
    {
    public:
        using value_type = number;
        using VectorType  = dealii::LinearAlgebra::distributed::Vector<number>;

        LaplaceOperator();
        virtual void clear() override;
        virtual void compute_diagonal() override;
        void vmult_unconstrained(VectorType &dst, const VectorType &src) const;

    private:
        virtual void apply_add(VectorType &dst, const VectorType &src) const override;

        void local_apply(const dealii::MatrixFree<dim, number>       &data,
                          VectorType                                  &dst,
                          const VectorType                            &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;
        
        void local_apply_plain(const dealii::MatrixFree<dim, number>       &data,
                            VectorType                                  &dst,
                            const VectorType                            &src,
                            const std::pair<unsigned int, unsigned int> &cell_range) const;

        void local_compute_diagonal(
            const dealii::MatrixFree<dim, number>       &data,
            VectorType                                  &dst,
            const unsigned int                          &dummy,
            const std::pair<unsigned int, unsigned int> &cell_range) const;
    };

    // Note on constraints: distribute_local_to_global(matrix, rhs, ...) is
    // not available here since there is no assembled matrix to pull the
    // -A_local*g inhomogeneous-boundary correction from. Hanging-node and
    // boundary constraints are instead enforced via:
    //   1. MatrixFree::reinit(..., constraints, ...), which makes
    //      LaplaceOperator::vmult respect constraints during apply_add
    //      (hanging-node coupling handled automatically, matching the
    //      assembled version's behavior for the homogeneous operator).
    //   2. An explicit inhomogeneity_correction vector holding the boundary
    //      values, subtracted from the RHS via a vmult (see assemble_system),
    //      then added back to the solution after CG (see solve). This
    //      reproduces exactly what distribute_local_to_global(matrix, rhs, ...)
    //      does implicitly when a matrix is available.
    
    template <int dim, int fe_degree>
    class MatrixFreePoissonSolver
        : public PoissonSolverBase<dim, dealii::LinearAlgebra::distributed::Vector<double>>
    {
    public:
        MatrixFreePoissonSolver(int degree, const Physics::AtomSystem<dim> &atom_system);

        void setup_system(int n_cells_per_edge,
                           double start,
                           double end,
                           const dealii::Function<dim> &boundary_condition) override;

        void assemble_system(const dealii::Function<dim> &forcing_function) override;
        int  solve(bool verbose) override;
        void output_results(const std::string &filename, bool verbose) const override;

        const dealii::LinearAlgebra::distributed::Vector<double> &get_solution() const override;
        const dealii::LinearAlgebra::distributed::Vector<double> &get_rhs() const override;

    private:
        using SystemMatrixType = LaplaceOperator<dim, fe_degree, double>;

        const Physics::AtomSystem<dim> &atom_system;

        std::shared_ptr<dealii::MatrixFree<dim, double>> mf_storage;
        SystemMatrixType                                  system_matrix;

        dealii::LinearAlgebra::distributed::Vector<double> solution;
        dealii::LinearAlgebra::distributed::Vector<double> system_rhs;
        dealii::LinearAlgebra::distributed::Vector<double> system_rhs_before;

        // Holds the inhomogeneous Dirichlet boundary values on constrained
        // DOFs (zero elsewhere) — the matrix-free analogue of the
        // -A_local*g correction folded in automatically by
        // distribute_local_to_global() when a matrix is available.
        dealii::LinearAlgebra::distributed::Vector<double> inhomogeneity_correction;
    };
}

#endif