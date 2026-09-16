#include "fd_preconditioner.h"
#include "laplace_operator.h"

#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

namespace Numerics
{
using namespace dealii;

// ============================================================================
// Free helper functions (same structure as the reference FastDiagonalization
// code): dense tensor-axis contractions and the FD solve itself, operating on
// a purely local, serial, size-N^3 dealii::Vector<double>.
// ============================================================================
namespace
{
    int fe_degree = 3; // TODO: pass this in from the preconditioner, or make it a template param

    void apply_axis_0(const LAPACKFullMatrix<double> &A, const Vector<double> &in,
                       Vector<double> &out, unsigned int N, bool transpose = false)
    {
        out = 0.0;
        for (unsigned int k = 0; k < N; ++k)
            for (unsigned int j = 0; j < N; ++j)
                for (unsigned int i = 0; i < N; ++i)
                {
                    double sum = 0.0;
                    for (unsigned int m = 0; m < N; ++m)
                        sum += (transpose ? A(m, i) : A(i, m)) * in(m + j * N + k * N * N);
                    out(i + j * N + k * N * N) = sum;
                }
    }

    void apply_axis_1(const LAPACKFullMatrix<double> &A, const Vector<double> &in,
                       Vector<double> &out, unsigned int N, bool transpose = false)
    {
        out = 0.0;
        for (unsigned int k = 0; k < N; ++k)
            for (unsigned int j = 0; j < N; ++j)
                for (unsigned int i = 0; i < N; ++i)
                {
                    double sum = 0.0;
                    for (unsigned int m = 0; m < N; ++m)
                        sum += (transpose ? A(m, j) : A(j, m)) * in(i + m * N + k * N * N);
                    out(i + j * N + k * N * N) = sum;
                }
    }

    void apply_axis_2(const LAPACKFullMatrix<double> &A, const Vector<double> &in,
                       Vector<double> &out, unsigned int N, bool transpose = false)
    {
        out = 0.0;
        for (unsigned int k = 0; k < N; ++k)
            for (unsigned int j = 0; j < N; ++j)
                for (unsigned int i = 0; i < N; ++i)
                {
                    double sum = 0.0;
                    for (unsigned int m = 0; m < N; ++m)
                        sum += (transpose ? A(m, k) : A(k, m)) * in(i + j * N + m * N * N);
                    out(i + j * N + k * N * N) = sum;
                }
    }

    // Solves (K⊗M⊗M + M⊗K⊗M + M⊗M⊗K) u = f via fast diagonalization,
    // given the precomputed generalized eigendecomposition (S, lambda) of
    // the 1D interior pencil (K_int, M_int).
    void fdm_solve(const Vector<double> &f_int, Vector<double> &u_int, unsigned int N,
                   const LAPACKFullMatrix<double> &S, const Vector<double> &lambda)
    {
        Vector<double> temp1(N * N * N), temp2(N * N * N), F_tilde(N * N * N);

        apply_axis_0(S, f_int, temp1, N, /*transpose=*/true);
        apply_axis_1(S, temp1, temp2, N, /*transpose=*/true);
        apply_axis_2(S, temp2, F_tilde, N, /*transpose=*/true);

        Vector<double> U_tilde(N * N * N);
        for (unsigned int k = 0; k < N; ++k)
            for (unsigned int j = 0; j < N; ++j)
                for (unsigned int i = 0; i < N; ++i)
                {
                    const unsigned int idx   = i + j * N + k * N * N;
                    const double       denom = lambda[i] + lambda[j] + lambda[k];
                    U_tilde[idx] = (std::abs(denom) > 1e-12) ? F_tilde[idx] / denom : 0.0;
                }

        u_int.reinit(N * N * N);
        apply_axis_2(S, U_tilde, temp1, N);
        apply_axis_1(S, temp1, temp2, N);
        apply_axis_0(S, temp2, u_int, N);
    }

    void solve_generalized_eigenproblem(LAPACKFullMatrix<double> &K,
                                         LAPACKFullMatrix<double> &M,
                                         Vector<double> &eigenvalues_out,
                                         LAPACKFullMatrix<double> &S_out)
    {
        const unsigned int           n = K.m();
        std::vector<Vector<double>>  eigenvectors;
        eigenvalues_out.reinit(n);

        K.compute_generalized_eigenvalues_symmetric(
            M,
            -std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            0.0,
            eigenvalues_out,
            eigenvectors,
            1);

        S_out.reinit(n, n);
        for (unsigned int j = 0; j < eigenvectors.size(); ++j)
            for (unsigned int i = 0; i < n; ++i)
                S_out(i, j) = eigenvectors[j][i];
    }
} // namespace

template <int dim, typename SystemMatrixType>
void FDPreconditioner<dim, SystemMatrixType>::initialize(
    const SystemMatrixType &system_matrix,
    const VectorType &system_rhs,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<double> &constraints,
    std::shared_ptr<const MatrixFree<dim, double>> mf_storage,
    int n_cells_per_edge)
{
    this->system_matrix_ptr = &system_matrix;
    this->system_rhs_ptr    = &system_rhs;
    this->dof_handler_ptr   = &dof_handler;
    this->constraints_ptr   = &constraints;
    this->mf_storage         = mf_storage;
    this->n_cells_per_edge   = n_cells_per_edge;

    if (this->mf_storage)
    {
        this->mf_storage->initialize_dof_vector(temp_vector);
        this->mf_storage->initialize_dof_vector(inv_diagonal);

        // Pull the already-computed diagonal inverse out of system_matrix,
        // rather than leaving inv_diagonal as an allocated-but-zero vector.
        inv_diagonal = system_matrix.get_matrix_diagonal_inverse()->get_vector();
    }

    setup_preconditioner_data();
}

template <int dim, typename SystemMatrixType>
void FDPreconditioner<dim, SystemMatrixType>::build_coarse_fd_system()
{
    static_assert(dim == 3, "FDPreconditioner coarse-grid FD system written for dim == 3.");

    const auto &triangulation = dof_handler_ptr->get_triangulation();

    // Domain extent. Assumes a cube domain (as in setup_system's
    // subdivided_hyper_cube); we don't have start/end passed into initialize(),
    // so we recover it from the mesh's bounding box.
    const auto bbox        = GridTools::compute_bounding_box(triangulation);
    const double domain_start = bbox.get_boundary_points().first[0];
    const double domain_end   = bbox.get_boundary_points().second[0];

    n_cells_per_edge_coarse = static_cast<unsigned int>(n_cells_per_edge);
    n_dofs_1d_coarse        = fe_degree * n_cells_per_edge_coarse + 1;
    n_int_coarse            = n_dofs_1d_coarse - 2;

    const QGauss<1>   quadrature_formula(fe_degree + 1);
    const FE_Q<1>     fe_1d(fe_degree);
    const unsigned int dofs_per_cell_1d = fe_1d.dofs_per_cell;

    const std::vector<unsigned int> lex_to_hierarchic =
        FETools::lexicographic_to_hierarchic_numbering<1>(fe_degree);

    FullMatrix<double> cell_K_1d(dofs_per_cell_1d, dofs_per_cell_1d);
    FullMatrix<double> cell_M_1d(dofs_per_cell_1d, dofs_per_cell_1d);

    const double h = (domain_end - domain_start) / static_cast<double>(n_cells_per_edge_coarse);

    for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
    {
        const Point<1> p_q = quadrature_formula.point(q);
        const double   w_q = quadrature_formula.weight(q);

        for (unsigned int i = 0; i < dofs_per_cell_1d; ++i)
        {
            const double phi_i  = fe_1d.shape_value(i, p_q);
            const double dphi_i = fe_1d.shape_grad(i, p_q)[0];

            for (unsigned int j = 0; j < dofs_per_cell_1d; ++j)
            {
                const double phi_j  = fe_1d.shape_value(j, p_q);
                const double dphi_j = fe_1d.shape_grad(j, p_q)[0];

                cell_K_1d(i, j) += dphi_i * (1.0 / h) * dphi_j * (1.0 / h) * w_q * h;
                cell_M_1d(i, j) += phi_i * phi_j * w_q * h;
            }
        }
    }

    K_1d_coarse.reinit(n_dofs_1d_coarse, n_dofs_1d_coarse);
    M_1d_coarse.reinit(n_dofs_1d_coarse, n_dofs_1d_coarse);
    K_1d_coarse = 0.0;
    M_1d_coarse = 0.0;

    for (unsigned int e = 0; e < n_cells_per_edge_coarse; ++e)
    {
        const unsigned int offset = e * fe_degree;
        for (unsigned int i_lex = 0; i_lex < dofs_per_cell_1d; ++i_lex)
        {
            const unsigned int i_hier = lex_to_hierarchic[i_lex];
            for (unsigned int j_lex = 0; j_lex < dofs_per_cell_1d; ++j_lex)
            {
                const unsigned int j_hier = lex_to_hierarchic[j_lex];
                K_1d_coarse(offset + i_lex, offset + j_lex) += cell_K_1d(i_hier, j_hier);
                M_1d_coarse(offset + i_lex, offset + j_lex) += cell_M_1d(i_hier, j_hier);
            }
        }
    }

    // Interior-only pencil -> generalized eigendecomposition (fast diagonalization).
    LAPACKFullMatrix<double> K_int(n_int_coarse, n_int_coarse);
    LAPACKFullMatrix<double> M_int(n_int_coarse, n_int_coarse);
    for (unsigned int i = 0; i < n_int_coarse; ++i)
        for (unsigned int j = 0; j < n_int_coarse; ++j)
        {
            K_int(i, j) = K_1d_coarse(i + 1, j + 1);
            M_int(i, j) = M_1d_coarse(i + 1, j + 1);
        }

    solve_generalized_eigenproblem(K_int, M_int, eigenvalues_int_coarse, S_1d_int_coarse);
}


template <int dim, typename SystemMatrixType>
void FDPreconditioner<dim, SystemMatrixType>::build_coarse_to_fine_map()
{
    static_assert(dim == 3, "FDPreconditioner coarse-grid mapping written for dim == 3.");

    const auto &triangulation = dof_handler_ptr->get_triangulation();
    const auto  bbox          = GridTools::compute_bounding_box(triangulation);
    const double domain_start = bbox.get_boundary_points().first[0];
    const double h_cell = (bbox.get_boundary_points().second[0] - domain_start) /
                          static_cast<double>(n_cells_per_edge_coarse);

    const std::vector<unsigned int> lex_to_hier =
        FETools::lexicographic_to_hierarchic_numbering<3>(fe_degree);

    const FiniteElement<dim> &fe = dof_handler_ptr->get_fe();
    std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);

    const unsigned int n3 = n_dofs_1d_coarse * n_dofs_1d_coarse * n_dofs_1d_coarse;
    std::vector<bool> valid(n3, false);
    std::vector<bool> contaminated(n3, false); // touched by a cell with a refined neighbor
    std::vector<types::global_dof_index> coarse_lex_to_fine_dof(
        n3, numbers::invalid_dof_index);

    for (const auto &cell : dof_handler_ptr->active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        if (cell->level() != 0)
            continue;

        // A level-0 cell is "safe" only if every face neighbor is also
        // unrefined (level 0 and active). Any face with a refined neighbor
        // means the true assembled matrix row for DOFs on that face carries
        // hanging-node contributions build_coarse_fd_system()'s from-scratch
        // 1D tensor assembly cannot see.
        bool cell_is_safe = true;
        for (const unsigned int face : cell->face_indices())
        {
            if (cell->at_boundary(face))
                continue; // domain boundary, not a refinement interface

            const auto neighbor = cell->neighbor(face);
            if (neighbor->has_children() || neighbor->level() != 0)
            {
                cell_is_safe = false;
                break;
            }
        }

        cell->get_dof_indices(local_dofs);

        const Point<dim> center = cell->center();
        const unsigned int cx =
            static_cast<unsigned int>(std::round((center[0] - domain_start - 0.5 * h_cell) / h_cell));
        const unsigned int cy =
            static_cast<unsigned int>(std::round((center[1] - domain_start - 0.5 * h_cell) / h_cell));
        const unsigned int cz =
            static_cast<unsigned int>(std::round((center[2] - domain_start - 0.5 * h_cell) / h_cell));

        for (unsigned int lx = 0; lx <= fe_degree; ++lx)
            for (unsigned int ly = 0; ly <= fe_degree; ++ly)
                for (unsigned int lz = 0; lz <= fe_degree; ++lz)
                {
                    const unsigned int local_lex =
                        lx + ly * (fe_degree + 1) + lz * (fe_degree + 1) * (fe_degree + 1);
                    const unsigned int local_hier = lex_to_hier[local_lex];

                    const unsigned int gx = cx * fe_degree + lx;
                    const unsigned int gy = cy * fe_degree + ly;
                    const unsigned int gz = cz * fe_degree + lz;

                    const unsigned int global_lex =
                        gx + gy * n_dofs_1d_coarse + gz * n_dofs_1d_coarse * n_dofs_1d_coarse;

                    coarse_lex_to_fine_dof[global_lex] = local_dofs[local_hier];
                    valid[global_lex]                  = true;

                    if (!cell_is_safe)
                        contaminated[global_lex] = true;
                }
    }

    // If ANY interior DOF is contaminated by a refined neighbor, disable FD
    // globally for this solve rather than feeding partially-zeroed forcing
    // into fdm_solve's dense, fully-coupled global solve — that pollutes
    // clean DOFs too (empirically confirmed: uniform mesh converges in one
    // iteration, but masking only the contaminated write-back while still
    // solving the full N^3 system degrades convergence badly).
    bool any_contaminated = false;
    for (unsigned int k = 0; k < n_int_coarse && !any_contaminated; ++k)
        for (unsigned int j = 0; j < n_int_coarse && !any_contaminated; ++j)
            for (unsigned int i = 0; i < n_int_coarse && !any_contaminated; ++i)
            {
                const unsigned int gx = i + 1, gy = j + 1, gz = k + 1;
                const unsigned int global_lex =
                    gx + gy * n_dofs_1d_coarse + gz * n_dofs_1d_coarse * n_dofs_1d_coarse;
                if (valid[global_lex] && contaminated[global_lex])
                    any_contaminated = true;
            }

    active_dof_to_lex_int.clear();

    if (any_contaminated)
        return; // stays empty -> vmult() falls back to pure Jacobi

    active_dof_to_lex_int.reserve(n_int_coarse * n_int_coarse * n_int_coarse);
    for (unsigned int k = 0; k < n_int_coarse; ++k)
        for (unsigned int j = 0; j < n_int_coarse; ++j)
            for (unsigned int i = 0; i < n_int_coarse; ++i)
            {
                const unsigned int gx = i + 1, gy = j + 1, gz = k + 1;
                const unsigned int global_lex =
                    gx + gy * n_dofs_1d_coarse + gz * n_dofs_1d_coarse * n_dofs_1d_coarse;
                if (!valid[global_lex])
                    continue;
                const unsigned int lex_int = i + j * n_int_coarse + k * n_int_coarse * n_int_coarse;
                active_dof_to_lex_int.emplace_back(coarse_lex_to_fine_dof[global_lex], lex_int);
            }
}


template <int dim, typename SystemMatrixType>
void FDPreconditioner<dim, SystemMatrixType>::setup_preconditioner_data()
{
    // TODO: currently assumes a single MPI rank (global vector element access
    // in vmult() below is not ghost/partition-aware). Extend once we've
    // confirmed the naive direct-mapping approach actually helps iteration
    // counts.
    build_coarse_fd_system();
    build_coarse_to_fine_map();
}

template <int dim, typename SystemMatrixType>
void FDPreconditioner<dim, SystemMatrixType>::vmult(VectorType &dst, const VectorType &src) const
{
    // 1. Jacobi baseline everywhere
    dst = src;
    dst.scale(inv_diagonal);

    if (!active_dof_to_lex_int.empty())
    {
        // 2. Residual after Jacobi: r = src - A * dst
        temp_vector = 0.0;
        system_matrix_ptr->vmult(temp_vector, dst);
        VectorType residual = src;
        residual -= temp_vector;

        // 3. Restrict residual to FD-active interior dofs
        Vector<double> f_int(n_int_coarse * n_int_coarse * n_int_coarse);
        f_int = 0.0;
        for (const auto &[fine_dof, lex_int] : active_dof_to_lex_int)
            f_int[lex_int] = residual(fine_dof);

        // 4. FD solve for the correction
        Vector<double> u_int;
        fdm_solve(f_int, u_int, n_int_coarse, S_1d_int_coarse, eigenvalues_int_coarse);

        // 5. Additive correction — NOT overwrite
        for (const auto &[fine_dof, lex_int] : active_dof_to_lex_int)
            dst(fine_dof) += u_int[lex_int];
    }

    if (constraints_ptr)
        constraints_ptr->set_zero(dst);
}

template class FDPreconditioner<3, LaplaceOperator<3, 1, double>>;
template class FDPreconditioner<3, LaplaceOperator<3, 2, double>>;
template class FDPreconditioner<3, LaplaceOperator<3, 3, double>>;
template class FDPreconditioner<3, LaplaceOperator<3, 4, double>>;
template class FDPreconditioner<3, LaplaceOperator<3, 5, double>>;
} // namespace Numerics