#include "laplace_operator_gpu.h"

namespace Numerics
{
 // Pre-computes gradients of basis functions at quadrature points 
 // Stores gradients in the shared memory of the GPU
  template <int dim, int fe_degree, typename number>
  DEAL_II_HOST_DEVICE void
  LaplaceOperatorGPUQuad<dim, fe_degree, number>::operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> *fe_eval,
    const int q_point) const
  {
    fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
  }

  // Operator
  template <int dim, int fe_degree, typename number>
  DEAL_II_HOST_DEVICE void
  LocalLaplaceOperatorGPU<dim, fe_degree, number>::operator()(
    const typename Portable::MatrixFree<dim, number>::Data *data,
    const Portable::DeviceVector<number>                    &src,
    Portable::DeviceVector<number>                          &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(data);

    // Reads src for given cell, evaluates gradients, and writes to dst
    // Calculates: u = sum_j src_j * phi_j
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);

    LaplaceOperatorGPUQuad<dim, fe_degree, number> quad;
    data->for_each_quad_point([&](const int &q_point) { quad(&fe_eval, q_point); });

    fe_eval.integrate(EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }

  // TODO
  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::reinit(
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<number> &constraints)
  {
    const MappingQ<dim> mapping(fe_degree);
    typename Portable::MatrixFree<dim, number>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                            update_JxW_values | update_quadrature_points;
    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);
  }

  // Implicitly applies the Laplace operator on the GPU: dst = A * src
  // Each cell is processed by a single GPU thread BLOCK 
  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.;
    LocalLaplaceOperatorGPU<dim, fe_degree, number> op;
    mf_data.cell_loop(op, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }

  // Only difference from vmult is the copy_constrained_values call, which is omitted here
  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::vmult_unconstrained(VectorType &dst, const VectorType &src) const
  {
    dst = 0.;
    LocalLaplaceOperatorGPU<dim, fe_degree, number> op;
    mf_data.cell_loop(op, src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::copy_constrained_values(const VectorType &src, VectorType &dst) const
  {
    mf_data.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::initialize_dof_vector(VectorType &vec) const
  {
    mf_data.initialize_dof_vector(vec);
  }

  // Computes the diagonal of the Laplace operator and stores its inverse in inverse_diagonal_entries
  // Used for preconditioning in iterative solvers
  template <int dim, int fe_degree, typename number>
  void LaplaceOperatorGPU<dim, fe_degree, number>::compute_diagonal()
  {
    inverse_diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    LaplaceOperatorGPUQuad<dim, fe_degree, number> quad;
    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1, number>(
      mf_data, inverse_diagonal, quad, EvaluationFlags::gradients, EvaluationFlags::gradients);

    number *raw_diagonal = inverse_diagonal.get_values();
    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

  // Explicit Instantiations
  
  template struct LaplaceOperatorGPUQuad<3, 1, double>;
  template struct LaplaceOperatorGPUQuad<3, 2, double>;
  template struct LaplaceOperatorGPUQuad<3, 3, double>;
  template struct LaplaceOperatorGPUQuad<3, 4, double>;
  template struct LaplaceOperatorGPUQuad<3, 5, double>;

  template struct LocalLaplaceOperatorGPU<3, 1, double>;
  template struct LocalLaplaceOperatorGPU<3, 2, double>;
  template struct LocalLaplaceOperatorGPU<3, 3, double>;
  template struct LocalLaplaceOperatorGPU<3, 4, double>;
  template struct LocalLaplaceOperatorGPU<3, 5, double>;

  template class LaplaceOperatorGPU<3, 1, double>;
  template class LaplaceOperatorGPU<3, 2, double>;
  template class LaplaceOperatorGPU<3, 3, double>;
  template class LaplaceOperatorGPU<3, 4, double>;
  template class LaplaceOperatorGPU<3, 5, double>;
}