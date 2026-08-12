#ifndef NUMERICS_MATRIX_FREE_POISSON_SOLVER_H
#define NUMERICS_MATRIX_FREE_POISSON_SOLVER_H

#include "poisson_solver_base.h"
#include "../physics/atom_system.h"
#include "laplace_operator.h"

namespace Numerics
{
    template <int dim, int fe_degree>
    class MatrixFreePoissonSolver
        : public PoissonSolverBase<dim>
    {
    public:
        MatrixFreePoissonSolver(int degree, const Physics::AtomSystem<dim> &atom_system);

        void setup_system(int n_cells_per_edge,
                          double start,
                          double end,
                          const dealii::Function<dim> &boundary_condition) override;

        // Overloaded helper if custom quadrature order is needed internally
        void assemble_system(const dealii::Function<dim> &forcing_function) override;

        int  solve(bool verbose) override;
        void output_results(const std::string &filename, bool verbose) const override;

        const HostVector &get_solution_host() const override;
        const HostVector &get_rhs() const;

    private:
        using SystemMatrixType = LaplaceOperator<dim, fe_degree, double>;

        const Physics::AtomSystem<dim> &atom_system;

        std::shared_ptr<dealii::MatrixFree<dim, double>> mf_storage;
        SystemMatrixType                                  system_matrix;

        HostVector solution;
        HostVector system_rhs;
        HostVector system_rhs_before;
        HostVector inhomogeneity_correction;
    };
}

#endif // NUMERICS_MATRIX_FREE_POISSON_SOLVER_H