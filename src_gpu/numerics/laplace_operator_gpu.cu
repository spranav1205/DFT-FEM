#include "laplace_operator_gpu.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <Kokkos_Core.hpp>

namespace Numerics
{
using namespace dealii;

// ================= LocalLaplaceOperatorGPU =================

template <int dim, int fe_degree, typename number>
DEAL_II_HOST_DEVICE void
LocalLaplaceOperatorGPU<dim, fe_degree, number>::operator()(
    const unsigned int                                     cell,
    const typename Portable::MatrixFree<dim, number>::Data *gpu_data,
    Portable::SharedData<dim, number>                       *shared_data,
    const number                                            *src,
    number                                                  *dst) const
{
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(
        gpu_data, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);

    for (int q = 0; q < fe_eval.n_q_points; ++q)
        fe_eval.submit_gradient(fe_eval.get_gradient(q), q);

    fe_eval.integrate(EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
}

// ================= LaplaceOperatorGPU =================

template <int dim, int fe_degree, typename number>
LaplaceOperatorGPU<dim, fe_degree, number>::LaplaceOperatorGPU()
{}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::clear()
{
    portable_mf = Portable::MatrixFree<dim, number>();
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::setup(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>            &dof_handler,
    const AffineConstraints<number>  &constraints,
    const Quadrature<1>               &quad)
{
    clear();

    typename Portable::MatrixFree<dim, number>::AdditionalData additional_data;
    // FIX: Portable::FEEvaluation::evaluate(EvaluationFlags::gradients)
    // still needs the shape-VALUE tables populated by the mapping to do its
    // tensor-product sum-factorization pass internally, even though only the
    // gradient is read back afterward. This differs from the host
    // FEEvaluation, where evaluate() only touches what's asked for. Omitting
    // update_values here was silently producing wrong (or garbage/zero)
    // results on the GPU path -- this was the source of the CPU/GPU vmult
    // mismatch.
    additional_data.mapping_update_flags =
        update_values | update_gradients | update_JxW_values;

    portable_mf.reinit(mapping, dof_handler, constraints, quad, additional_data);

    std::cout << "LaplaceOperatorGPU setup: cells=" << dof_handler.get_triangulation().n_active_cells()
               << ", dofs/cell=" << dof_handler.get_fe().n_dofs_per_cell() << std::endl;
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::initialize_dof_vector(VectorType &vec) const
{
    portable_mf.initialize_dof_vector(vec);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::vmult(VectorType &dst, const VectorType &src) const
{
    dst = 0;

    LocalLaplaceOperatorGPU<dim, fe_degree, number> local_operator;
    portable_mf.cell_loop(local_operator, src, dst);

    // NOTE: cell_loop on the device is asynchronous with respect to the host;
    // copy_constrained_values below reads/writes the same VectorType, and
    // deal.II's Portable::MatrixFree::cell_loop is documented to synchronize
    // internally before returning. If you ever see nondeterministic
    // (run-to-run varying) mismatches rather than a consistent offset, add
    // Kokkos::fence() here as a diagnostic -- that would point to a missing
    // synchronization rather than a numerics bug.
    portable_mf.copy_constrained_values(src, dst);
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::vmult_add(VectorType &dst, const VectorType &src) const
{
    VectorType tmp;
    initialize_dof_vector(tmp);
    vmult(tmp, src);
    dst += tmp;
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::compute_inverse_diagonal(VectorType &inverse_diagonal) const
{
    // STUB: still returns all-ones (no real Jacobi scaling). Do not trust
    // Jacobi-preconditioned CG results on this GPU path until this is
    // replaced with a real per-DOF diagonal probe. Left unchanged here since
    // it is unrelated to the vmult mismatch -- flagging again so it isn't
    // forgotten once vmult is verified to match.
    initialize_dof_vector(inverse_diagonal);
    inverse_diagonal = 1.0;
}

template <int dim, int fe_degree, typename number>
void LaplaceOperatorGPU<dim, fe_degree, number>::vmult_unconstrained(
    VectorType &dst, const VectorType &src) const
{
    dst = 0;
    LocalLaplaceOperatorGPU<dim, fe_degree, number> local_operator;
    portable_mf.cell_loop(local_operator, src, dst);
}

// Explicit template instantiations
template class LaplaceOperatorGPU<3, 1, double>;
template class LaplaceOperatorGPU<3, 2, double>;
template class LaplaceOperatorGPU<3, 3, double>;
template class LaplaceOperatorGPU<3, 4, double>;
template class LaplaceOperatorGPU<3, 5, double>;
}