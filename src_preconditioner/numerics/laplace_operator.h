#ifndef NUMERICS_LAPLACE_OPERATOR_H
#define NUMERICS_LAPLACE_OPERATOR_H

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
}

#endif