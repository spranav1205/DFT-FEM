#include "laplace_operator.h"

#include <cmath>

namespace Numerics
{
    using namespace dealii;

    template <int dim, int fe_degree, typename number>
    LaplaceOperator<dim, fe_degree, number>::LaplaceOperator()
        : MatrixFreeOperators::Base<dim, VectorType>()
    {}

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::clear()
    {
        MatrixFreeOperators::Base<dim, VectorType>::clear();
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_apply(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const VectorType                            &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);

            // Read local coefficients u_j for this cell and hanging-node constraints.
            phi.read_dof_values(src);

            // Interpolate the local DoFs to compute ∇u_h at each quadrature point.
            phi.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
                // Store the flux F = ∇u_h at each quadrature point.
                phi.submit_gradient(phi.get_gradient(q), q);

            // Compute (A_local u_local)_i = ∫_K ∇φ_i · F dx.
            phi.integrate(EvaluationFlags::gradients);

            // Add the local operator result into the global vector.
            phi.distribute_local_to_global(dst);
        }
    }


    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::apply_add(VectorType &dst, const VectorType &src) const
    {
        this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_compute_diagonal(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const unsigned int                          &,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
        AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                    phi.submit_dof_value(VectorizedArray<number>(), j);
                phi.submit_dof_value(make_vectorized_array<number>(1.0), i);

                phi.evaluate(EvaluationFlags::gradients);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                    phi.submit_gradient(phi.get_gradient(q), q);
                phi.integrate(EvaluationFlags::gradients);

                diagonal[i] = phi.get_dof_value(i);
            }

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
                phi.submit_dof_value(diagonal[i], i);

            phi.distribute_local_to_global(dst);
        }
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
    {
        this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
        VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
        this->data->initialize_dof_vector(inverse_diagonal);

        unsigned int dummy = 0;
        this->data->cell_loop(&LaplaceOperator::local_compute_diagonal, this, inverse_diagonal, dummy);

        // Forces constrained (hanging-node/boundary) DOFs to have diagonal 1,
        // matching set_constrained_entries_to_one's role for the assembled
        // matrix: CG never needs to "solve" for these rows since their value
        // is fixed by constraints.distribute() afterward.
        this->set_constrained_entries_to_one(inverse_diagonal);

        for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
        {
            const double val = inverse_diagonal.local_element(i);
            inverse_diagonal.local_element(i) = (std::abs(val) > 1e-15) ? 1.0 / val : 1.0;
        }
    }

    // numerics/matrix_free_poisson_solver.cpp

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::local_apply_plain(
        const MatrixFree<dim, number>               &data,
        VectorType                                  &dst,
        const VectorType                            &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
            phi.reinit(cell);
            phi.read_dof_values_plain(src); // raw read: no constraint zeroing
            phi.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate(EvaluationFlags::gradients);
            phi.distribute_local_to_global(dst); // still respects constraints on write
        }
    }

    template <int dim, int fe_degree, typename number>
    void LaplaceOperator<dim, fe_degree, number>::vmult_unconstrained(
        VectorType &dst, const VectorType &src) const
    {
        dst = 0;
        this->data->cell_loop(&LaplaceOperator::local_apply_plain, this, dst, src);
        dst.compress(VectorOperation::add);
    }

    // Explicit template instantiations for the supported dimensions and FE degrees.
    template class LaplaceOperator<3, 1, double>;
    template class LaplaceOperator<3, 2, double>;
    template class LaplaceOperator<3, 3, double>;
    template class LaplaceOperator<3, 4, double>;
    template class LaplaceOperator<3, 5, double>;
}