#ifndef NUMERICS_MATRIX_FREE_POISSON_SOLVER_GPU_H
#define NUMERICS_MATRIX_FREE_POISSON_SOLVER_GPU_H

#include "poisson_solver_base.h"
#include "laplace_operator_gpu.h"

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/diagonal_matrix.h>

namespace Physics
{
    template <int dim>
    class AtomSystem;
}

namespace Numerics
{
    // GPU counterpart of MatrixFreePoissonSolver. Same public interface
    // (setup_system / assemble_system / solve / output_results / get_solution
    // / get_rhs) and same overall structure -- only VectorType and the
    // operator type differ: LaplaceOperatorGPU instead of LaplaceOperator,
    // and vectors living in MemorySpace::Default (device) instead of the
    // implicit host memory space.
    //
    // Mesh handling, DoFHandler, AffineConstraints, and RHS assembly remain
    // on the host exactly as in MatrixFreePoissonSolver -- only the
    // matrix-free apply (the CG matvec) runs on device. See
    // portable_laplace_operator_gpu.h for details.
    template <int dim, int fe_degree>
    class MatrixFreePoissonSolverGPU
        : public PoissonSolverBase<dim, dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default>>
    {
    public:
        using VectorType = dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Default>;
        using SystemMatrixType = LaplaceOperatorGPU<dim, fe_degree, double>;

        MatrixFreePoissonSolverGPU(int degree, const Physics::AtomSystem<dim> &atom_system);

        void setup_system(int n_cells_per_edge,
                           double start,
                           double end,
                           const dealii::Function<dim> &boundary_condition) override;

        void assemble_system(const dealii::Function<dim> &forcing_function) override;
        int  solve(bool verbose) override;
        void output_results(const std::string &filename, bool verbose) const override;

        const VectorType &get_solution() const override;
        const VectorType &get_rhs() const override;

    private:
        const Physics::AtomSystem<dim> &atom_system;

        SystemMatrixType system_matrix;

        VectorType solution;
        VectorType system_rhs;
        VectorType system_rhs_before;
        VectorType inhomogeneity_correction;
        VectorType inverse_diagonal; // built by compute_inverse_diagonal(); see .cu for current stub status
    };
}

#endif