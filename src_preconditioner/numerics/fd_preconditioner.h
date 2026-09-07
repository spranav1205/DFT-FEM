#ifndef FD_PRECONDITIONER_H
#define FD_PRECONDITIONER_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <memory>

namespace Numerics
{
    using namespace dealii;

    template <int dim, typename SystemMatrixType>
    class FDPreconditioner : public Subscriptor
    {
    public:
        using VectorType = LinearAlgebra::distributed::Vector<double>;

        FDPreconditioner() = default;

        /**
         * @brief Initialize preconditioner state and internal structures.
         * 
         * @param system_matrix Matrix-free operator A
         * @param system_rhs System right-hand side vector
         * @param dof_handler Degree of freedom handler (cell-to-dof mapping)
         * @param constraints Boundary and hanging-node constraints
         * @param mf_storage Shared matrix-free storage
         * @param n_cells_per_edge Grid discretization parameter
         */
        void initialize(const SystemMatrixType &system_matrix,
                        const VectorType &system_rhs,
                        const DoFHandler<dim> &dof_handler,
                        const AffineConstraints<double> &constraints,
                        std::shared_ptr<const MatrixFree<dim, double>> mf_storage,
                        int n_cells_per_edge);

        /**
         * @brief Applies preconditioner action: dst = M^-1 * src
         * 
         * @param dst Preconditioned result vector
         * @param src Current residual vector passed by SolverCG
         */
        void vmult(VectorType &dst, const VectorType &src) const;

    private:
        void setup_preconditioner_data();

        // Pointers & references to framework infrastructure
        const SystemMatrixType *system_matrix_ptr = nullptr;
        const VectorType *system_rhs_ptr          = nullptr;
        const DoFHandler<dim> *dof_handler_ptr    = nullptr;
        const AffineConstraints<double> *constraints_ptr = nullptr;
        std::shared_ptr<const MatrixFree<dim, double>> mf_storage;

        int n_cells_per_edge = 0;

        // Workspace vectors (mutable allows modification inside const vmult)
        mutable VectorType temp_vector;
        VectorType inv_diagonal;
    };
}
#endif // FD_PRECONDITIONER_H

