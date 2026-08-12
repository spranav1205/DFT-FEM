#ifndef NUMERICS_LAPLACE_OPERATOR_GPU_H
#define NUMERICS_LAPLACE_OPERATOR_GPU_H

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <Kokkos_Core.hpp>
#include <memory>

namespace Numerics
{
  using namespace dealii;

  // Data to store on the GPU for the matrix-free Laplace operator
  template <int dim, int fe_degree, typename number = double>
  struct LaplaceOperatorGPUQuad
  {
    DEAL_II_HOST_DEVICE void operator()(
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> *fe_eval,
      const int q_point) const;

    static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);
  };

  template <int dim, int fe_degree, typename number = double>
  struct LocalLaplaceOperatorGPU
  {
    DEAL_II_HOST_DEVICE void
    operator()(const typename Portable::MatrixFree<dim, number>::Data *data,
               const Portable::DeviceVector<number>                    &src,
               Portable::DeviceVector<number>                          &dst) const;
    static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);
  };

  template <int dim, int fe_degree, typename number = double>
  class LaplaceOperatorGPU : public EnableObserverPointer
  {
  public:
    using value_type = number;
    using VectorType = LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

    LaplaceOperatorGPU() = default;

    // Added constructor to match instantiation in matrix_free_poisson_solver_gpu.cpp
    LaplaceOperatorGPU(const DoFHandler<dim> &dof_handler,
                       const AffineConstraints<number> &constraints)
    {
      reinit(dof_handler, constraints);
    }

    void reinit(const DoFHandler<dim> &dof_handler,
                const AffineConstraints<number> &constraints);

    void vmult(VectorType &dst, const VectorType &src) const;

    void vmult_unconstrained(VectorType &dst, const VectorType &src) const;

    void copy_constrained_values(const VectorType &src, VectorType &dst) const;

    void initialize_dof_vector(VectorType &vec) const;

    void compute_diagonal();

    std::shared_ptr<DiagonalMatrix<VectorType>> get_matrix_diagonal_inverse() const
    {
      return inverse_diagonal_entries;
    }

    types::global_dof_index m() const { return mf_data.get_vector_partitioner()->size(); }
    types::global_dof_index n() const { return mf_data.get_vector_partitioner()->size(); }

  private:
    Portable::MatrixFree<dim, number> mf_data;
    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  };
}

#endif