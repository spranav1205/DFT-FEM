#ifndef NUMERICS_PORTABLE_LAPLACE_OPERATOR_GPU_H
#define NUMERICS_PORTABLE_LAPLACE_OPERATOR_GPU_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>

#include <Kokkos_Core.hpp>

namespace Numerics
{
    // Device functor: the actual per-cell Laplace action, evaluated on GPU
    // via Portable::FEEvaluation.
    //
    // Signature matches deal.II 9.6.2's actual Portable::MatrixFree
    // functor convention (see the internal ApplyKernel wrapper in
    // portable_matrix_free.templates.h, and the step-64 tutorial's
    // LocalHelmholtzOperator for the canonical example): the framework
    // calls func(cell, &gpu_data, &shared_data, src, dst) with src/dst
    // as raw pointers, NOT Kokkos::View.
    template <int dim, int fe_degree, typename number>
    class LocalLaplaceOperatorGPU
    {
    public:
        DEAL_II_HOST_DEVICE void
        operator()(
            const unsigned int                                               cell,
            const typename dealii::Portable::MatrixFree<dim, number>::Data *gpu_data,
            dealii::Portable::SharedData<dim, number>                       *shared_data,
            const number                                                    *src,
            number                                                          *dst) const;

        static const unsigned int n_local_dofs =
            dealii::Utilities::pow(fe_degree + 1, dim);
        static const unsigned int n_q_points =
            dealii::Utilities::pow(fe_degree + 1, dim);
    };

    // GPU counterpart of Numerics::LaplaceOperator (see laplace_operator.h).
    // NOTE: unlike the CPU operator, this class does NOT derive from
    // dealii::MatrixFreeOperators::Base. Base's internal ghost-range/
    // partitioner handling (adjust_ghost_range_if_necessary) assumes a host
    // dealii::MatrixFree object and MemorySpace::Host vectors -- it is not
    // compatible with device (MemorySpace::Default) vectors. This matches
    // step-64's own operator, which is likewise standalone (not Base-derived)
    // for the same reason.
    //
    // Consequently PreconditionJacobi<SystemMatrixType> (which relies on
    // Base's el()/compute_diagonal() machinery) cannot be used here either;
    // use dealii::DiagonalMatrix<VectorType> instead, built from
    // compute_inverse_diagonal()'s output.
    template <int dim, int fe_degree, typename number = double>
    class LaplaceOperatorGPU
    {
    public:
        using VectorType =
            dealii::LinearAlgebra::distributed::Vector<number, dealii::MemorySpace::Default>;

        LaplaceOperatorGPU();

        void clear();

        // Same name/signature convention as the CPU operator's setup(), plus
        // the constraints argument (needed so copy_constrained_values can
        // enforce identity behavior on constrained DOFs, since the device
        // functor has no notion of them).
        void setup(const dealii::Mapping<dim>               &mapping,
                   const dealii::DoFHandler<dim>             &dof_handler,
                   const dealii::AffineConstraints<number>   &constraints,
                   const dealii::Quadrature<1>                &quad);

        void initialize_dof_vector(VectorType &vec) const;

        // vmult/vmult_add: the interface SolverCG actually needs (it does
        // not require the full MatrixFreeOperators::Base surface).
        void vmult(VectorType &dst, const VectorType &src) const;
        void vmult_add(VectorType &dst, const VectorType &src) const;

        // Same name/purpose as before: apply the operator ignoring
        // constraints (used for the inhomogeneity correction in
        // MatrixFreePoissonSolverGPU::assemble_system).
        void vmult_unconstrained(VectorType &dst, const VectorType &src) const;

        // STUB, see .cu: currently returns an all-ones vector (i.e. no real
        // Jacobi scaling) until a device diagonal-probing functor is written
        // and verified against this deal.II version's Portable::FEEvaluation
        // dof-value API. Wrap the result in dealii::DiagonalMatrix<VectorType>
        // at the call site.
        void compute_inverse_diagonal(VectorType &inverse_diagonal) const;

    private:
        dealii::Portable::MatrixFree<dim, number> portable_mf;
    };

    template <int dim, int fe_degree, typename number = double>
    using LaplaceOperatorGPUAlias = LaplaceOperatorGPU<dim, fe_degree, number>;
}

#endif