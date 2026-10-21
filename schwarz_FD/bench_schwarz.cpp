
/*
 * bench_schwarz.cpp
 *
 * Standalone benchmark for a 3D, two-level-refined overlapping
 * Schwarz preconditioner using fast diagonalization (FD).
 *
 * Build with deal.II and an optimized compiler configuration.
 *
 * Examples:
 *   ./bench --help
 *   ./bench --n_coarse=16 --mid_first=4 --mid_count=8 \
 *           --inner_first=6 --inner_count=4 --overlap=1 \
 *           --warmup=10 --reps=200
 *
 * This benchmark measures vmult() only. Setup is excluded.
 * No correctness fingerprint, CG solve, STREAM test, or roofline model.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>

using namespace dealii;

// ============================================================================
// Parameters
// ============================================================================

struct Parameters
{
    unsigned int n_coarse    = 16;
    unsigned int mid_first   = 4;
    unsigned int mid_count   = 8;
    unsigned int inner_first = 6;
    unsigned int inner_count = 4;

    unsigned int overlap     = 1;
    unsigned int warmup      = 10;
    unsigned int reps        = 100;
    unsigned int seed        = 12345;
    std::string csv_path;

    static void print_help(const char *program)
    {
        std::cout
            << "Usage: " << program << " [options]\n\n"
            << "Mesh and refinement:\n"
            << "  --n_coarse=N       Coarse cells per dimension (16)\n"
            << "  --mid_first=N      First middle-region coarse cell (4)\n"
            << "  --mid_count=N      Middle-region width in coarse cells (8)\n"
            << "  --inner_first=N    First inner-region coarse cell (6)\n"
            << "  --inner_count=N    Inner-region width in coarse cells (4)\n\n"
            << "Preconditioner:\n"
            << "  --overlap=N        Patch extension in coarser cells (1)\n\n"
            << "Benchmark:\n"
            << "  --warmup=N         Untimed warm-up applications (10)\n"
            << "  --reps=N           Timed applications (200)\n"
            << "  --seed=N           Random input seed (12345)\n"
            << "  --csv=PATH        Write one benchmark row to CSV at PATH\n"
            << "  --help             Show this message\n";
    }

    static Parameters parse(int argc, char **argv)
    {
        Parameters p;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg(argv[i]);

            if (arg == "--help" || arg == "-h")
            {
                print_help(argv[0]);
                std::exit(0);
            }

            const auto eq = arg.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error(
                    "Expected --option=value, got: " + arg);

            const std::string key = arg.substr(0, eq);
            const std::string value = arg.substr(eq + 1);

            std::size_t consumed = 0;
            const unsigned long parsed = std::stoul(value, &consumed);

            if (consumed != value.size())
                throw std::runtime_error("Invalid integer: " + value);

            if (parsed > std::numeric_limits<unsigned int>::max())
                throw std::runtime_error("Integer too large: " + value);

            const auto v = static_cast<unsigned int>(parsed);

            if (key == "--n_coarse")
                p.n_coarse = v;
            else if (key == "--mid_first")
                p.mid_first = v;
            else if (key == "--mid_count")
                p.mid_count = v;
            else if (key == "--inner_first")
                p.inner_first = v;
            else if (key == "--inner_count")
                p.inner_count = v;
            else if (key == "--overlap")
                p.overlap = v;
            else if (key == "--warmup")
                p.warmup = v;
            else if (key == "--reps")
                p.reps = v;
            else if (key == "--seed")
                p.seed = v;
            else if (key == "--csv")
                p.csv_path = value;
            else
                throw std::runtime_error("Unknown option: " + key);
        }

        p.validate();
        return p;
    }

    void validate() const
    {
        if (n_coarse < 2)
            throw std::runtime_error("--n_coarse must be >= 2");

        if (mid_count == 0 || inner_count == 0)
            throw std::runtime_error("Refinement widths must be positive");

        if (mid_first + mid_count > n_coarse)
            throw std::runtime_error("Middle region lies outside the domain");

        if (inner_first < mid_first ||
            inner_first + inner_count > mid_first + mid_count)
            throw std::runtime_error(
                "Inner region must lie inside the middle region");

        if (reps == 0)
            throw std::runtime_error("--reps must be positive");
    }
};

struct Region
{
    double lo, hi;
};

// ============================================================================
// Two-level-refined overlapping Schwarz preconditioner
// ============================================================================

template <int dim>
class TwoFDOverlappingSchwarzPreconditioner
{
public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    struct Geometry
    {
        double domain_start;
        double domain_end;
        double coarse_h;
        unsigned int overlap_cells;
        std::vector<Region> layers;
    };

    void initialize(const MatrixFree<dim, double> &mf,
                    const AffineConstraints<double> &constraints,
                    const VectorType &inverse_diagonal,
                    const Geometry &geo);

    void vmult(VectorType &dst, const VectorType &src) const;

private:
    struct Eig1D
    {
        unsigned int n = 0;
        std::vector<double> S;
        std::vector<double> St;
        std::vector<double> lambda;
    };

    struct Patch
    {
        std::string name;
        std::array<unsigned int, 3> n_int{};
        std::array<std::shared_ptr<Eig1D>, 3> eig;

        // Patch lattice positions and corresponding vector-local indices.
        std::vector<unsigned int> lattice_idx;
        std::vector<unsigned int> local_idx;

        // Preallocated work buffers.
        mutable std::vector<double> loc, t1, t2;
    };

    std::shared_ptr<Eig1D> get_eig(unsigned int degree,
                                   unsigned int n_cells,
                                   double h);

    static void apply_axis(const std::vector<double> &mat,
                           unsigned int na,
                           const std::vector<double> &in,
                           std::vector<double> &out,
                           const std::array<unsigned int, 3> &n,
                           unsigned int axis);

    void fd_solve(const Patch &patch) const;

    std::map<std::pair<unsigned int, double>,
             std::shared_ptr<Eig1D>> eig_cache;

    std::vector<Patch> patches;

    // Uncovered unconstrained DoFs use Jacobi.
    std::vector<std::pair<unsigned int, double>> fallback_local;

    // Constrained DoFs use identity.
    std::vector<unsigned int> constrained_local;
};

// ----------------------------------------------------------------------------
// Construct 1D generalized eigenpairs K phi = lambda M phi
// ----------------------------------------------------------------------------

template <int dim>
std::shared_ptr<
    typename TwoFDOverlappingSchwarzPreconditioner<dim>::Eig1D>
TwoFDOverlappingSchwarzPreconditioner<dim>::get_eig(
    unsigned int degree,
    unsigned int n_cells,
    double h)
{
    const auto key = std::make_pair(n_cells, h);

    const auto it = eig_cache.find(key);
    if (it != eig_cache.end())
        return it->second;

    const unsigned int N = degree * n_cells + 1;
    const unsigned int n = N - 2;

    FE_Q<1> fe1(degree);
    QGauss<1> quad(degree + 1);

    const unsigned int dpc = fe1.n_dofs_per_cell();
    const std::vector<unsigned int> l2h =
        FETools::lexicographic_to_hierarchic_numbering<1>(degree);

    FullMatrix<double> cK(dpc, dpc), cM(dpc, dpc);

    for (unsigned int q = 0; q < quad.size(); ++q)
    {
        const Point<1> pq = quad.point(q);
        const double wq = quad.weight(q);

        for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
            {
                cK(i, j) +=
                    fe1.shape_grad(i, pq)[0] *
                    fe1.shape_grad(j, pq)[0] * wq / h;

                cM(i, j) +=
                    fe1.shape_value(i, pq) *
                    fe1.shape_value(j, pq) * wq * h;
            }
    }

    FullMatrix<double> K(N, N), M(N, N);

    for (unsigned int e = 0; e < n_cells; ++e)
        for (unsigned int il = 0; il < dpc; ++il)
            for (unsigned int jl = 0; jl < dpc; ++jl)
            {
                K(e * degree + il, e * degree + jl) +=
                    cK(l2h[il], l2h[jl]);

                M(e * degree + il, e * degree + jl) +=
                    cM(l2h[il], l2h[jl]);
            }

    LAPACKFullMatrix<double> K_int(n, n), M_int(n, n);

    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            K_int(i, j) = K(i + 1, j + 1);
            M_int(i, j) = M(i + 1, j + 1);
        }

    Vector<double> ev(n);
    std::vector<Vector<double>> evec;

    K_int.compute_generalized_eigenvalues_symmetric(
        M_int,
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        0.0,
        ev,
        evec,
        1);

    AssertThrow(evec.size() == n,
                ExcMessage("Unexpected number of eigenvectors"));

    auto e1 = std::make_shared<Eig1D>();

    e1->n = n;
    e1->S.assign(static_cast<std::size_t>(n) * n, 0.0);
    e1->St.assign(static_cast<std::size_t>(n) * n, 0.0);
    e1->lambda.resize(n);

    for (unsigned int j = 0; j < n; ++j)
    {
        e1->lambda[j] = ev[j];

        for (unsigned int i = 0; i < n; ++i)
        {
            e1->S[i * n + j] = evec[j][i];
            e1->St[j * n + i] = evec[j][i];
        }
    }

    eig_cache[key] = e1;
    return e1;
}

// ----------------------------------------------------------------------------
// Apply a dense 1D transform along one lattice axis.
// Lattice layout: i + j*n[0] + k*n[0]*n[1].
// ----------------------------------------------------------------------------
static inline void sum_axpy(const double *__restrict a, const unsigned nm,
                            const double *__restrict x, const std::size_t ld,
                            double *__restrict y, const std::size_t len)
{
    std::size_t p = 0;

    for (; p + 16 <= len; p += 16)
    {
        __m256d c0 = _mm256_setzero_pd(), c1 = c0, c2 = c0, c3 = c0;
        for (unsigned m = 0; m < nm; ++m)
        {
            const __m256d va = _mm256_set1_pd(a[m]);
            const double *xp = x + m * ld + p;
            c0 = _mm256_fmadd_pd(va, _mm256_loadu_pd(xp),      c0);
            c1 = _mm256_fmadd_pd(va, _mm256_loadu_pd(xp + 4),  c1);
            c2 = _mm256_fmadd_pd(va, _mm256_loadu_pd(xp + 8),  c2);
            c3 = _mm256_fmadd_pd(va, _mm256_loadu_pd(xp + 12), c3);
        }
        _mm256_storeu_pd(y + p,      c0);
        _mm256_storeu_pd(y + p + 4,  c1);
        _mm256_storeu_pd(y + p + 8,  c2);
        _mm256_storeu_pd(y + p + 12, c3);
    }
    for (; p + 4 <= len; p += 4)
    {
        __m256d c = _mm256_setzero_pd();
        for (unsigned m = 0; m < nm; ++m)
            c = _mm256_fmadd_pd(_mm256_set1_pd(a[m]),
                                _mm256_loadu_pd(x + m * ld + p), c);
        _mm256_storeu_pd(y + p, c);
    }
    for (; p < len; ++p)
    {
        double s = 0.0;
        for (unsigned m = 0; m < nm; ++m)
            s += a[m] * x[m * ld + p];
        y[p] = s;
    }
}

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::apply_axis(
    const std::vector<double> &mat,
    const unsigned int na,
    const std::vector<double> &in,
    std::vector<double> &out,
    const std::array<unsigned int, 3> &n,
    const unsigned int axis)
{
    const unsigned int n0 = n[0];
    const unsigned int n1 = n[1];
    const unsigned int n2 = n[2];

    if (axis == 0)
    {
        for (unsigned k = 0; k < n2; ++k)
            for (unsigned j = 0; j < n1; ++j)
            {
                const std::size_t off = std::size_t(j) * n0 + std::size_t(k) * n0 * n1;
                // a = input line (contiguous), "x" = matT rows
                sum_axpy(&in[off], na, mat.data(), na, &out[off], n0);
            }
    }
    else if (axis == 1)
    {
        for (unsigned k = 0; k < n2; ++k)
        {
            const std::size_t kbase = std::size_t(k) * n0 * n1;
            for (unsigned j = 0; j < n1; ++j)
                sum_axpy(&mat[std::size_t(j) * na], n1,
                        &in[kbase], n0, &out[kbase + std::size_t(j) * n0], n0);
        }
    }
    else
    {
        const std::size_t plane = std::size_t(n0) * n1;
        constexpr std::size_t PB = 128;               // tune: n2*PB*8 B should fit in L1
        for (std::size_t p0 = 0; p0 < plane; p0 += PB)
        {
            const std::size_t len = std::min(PB, plane - p0);
            for (unsigned k = 0; k < n2; ++k)
                sum_axpy(&mat[std::size_t(k) * na], n2,
                        &in[p0], plane, &out[std::size_t(k) * plane + p0], len);
        }
    }
}

// ----------------------------------------------------------------------------
// Fast diagonalization solve.
//
// Solves:
// (Kx x My x Mz + Mx x Ky x Mz + Mx x My x Kz) u = f
//
// The input and output are patch.loc.
// ----------------------------------------------------------------------------

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::fd_solve(
    const Patch &patch) const
{
    const auto &n = patch.n_int;
    auto &u = patch.loc;
    auto &t1 = patch.t1;
    auto &t2 = patch.t2;

    // NOTE: For data locality for apply_axis 0, we take the transpose of the first eigenvector matrix!

    apply_axis(patch.eig[0]->S, n[0], u,  t1, n, 0);
    apply_axis(patch.eig[1]->St, n[1], t1, t2, n, 1);
    apply_axis(patch.eig[2]->St, n[2], t2, t1, n, 2);

    for (unsigned int k = 0; k < n[2]; ++k)
        for (unsigned int j = 0; j < n[1]; ++j)
            for (unsigned int i = 0; i < n[0]; ++i)
            {
                const std::size_t idx =
                    i + static_cast<std::size_t>(j) * n[0] +
                    static_cast<std::size_t>(k) * n[0] * n[1];

                t1[idx] /=
                    patch.eig[0]->lambda[i] +
                    patch.eig[1]->lambda[j] +
                    patch.eig[2]->lambda[k];
            }

    apply_axis(patch.eig[2]->S, n[2], t1, t2, n, 2);
    apply_axis(patch.eig[1]->S, n[1], t2, t1, n, 1);
    apply_axis(patch.eig[0]->St, n[0], t1, u,  n, 0);
}

// ----------------------------------------------------------------------------
// Construct patches and assign ownership.
// ----------------------------------------------------------------------------

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::initialize(
    const MatrixFree<dim, double> &mf,
    const AffineConstraints<double> &constraints,
    const VectorType &inverse_diagonal,
    const Geometry &geo)
{
    static_assert(dim == 3, "Written for dim == 3.");

    const DoFHandler<dim> &dof_handler = mf.get_dof_handler();
    const auto &partitioner = mf.get_vector_partitioner();

    AssertThrow(partitioner->n_mpi_processes() == 1,
                ExcMessage("This preconditioner is serial only."));

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const unsigned int degree = fe.degree;
    const unsigned int dpc = fe.n_dofs_per_cell();
    const types::global_dof_index n_dofs = dof_handler.n_dofs();

    const unsigned int L = geo.layers.size();
    AssertThrow(L >= 1, ExcMessage("Need at least one layer"));

    const double finest_h =
        geo.coarse_h / (1u << (L - 1));

    const double tol = 1e-6 * finest_h;

    const std::vector<unsigned int> l2h =
        FETools::lexicographic_to_hierarchic_numbering<dim>(degree);

    AssertThrow(l2h.size() == dpc, ExcInternalError());

    struct Spec
    {
        std::string name;
        double lo, hi, h_patch;
    };

    // Finest layer first. Finer layers claim their DoFs first.
    std::vector<Spec> specs;

    for (unsigned int ll = 0; ll < L; ++ll)
    {
        const unsigned int l = L - 1 - ll;

        Spec s;
        s.h_patch = geo.coarse_h / (1u << l);

        if (l == 0)
        {
            s.lo = geo.domain_start;
            s.hi = geo.domain_end;
        }
        else
        {
            const double ext =
                geo.overlap_cells * geo.coarse_h / (1u << (l - 1));

            s.lo = std::max(geo.domain_start, geo.layers[l].lo - ext);
            s.hi = std::min(geo.domain_end, geo.layers[l].hi + ext);
        }

        if (L == 3)
            s.name = (l == 0 ? "outer" :
                      (l == 1 ? "middle" : "inner"));
        else
            s.name = "layer" + std::to_string(l);

        specs.push_back(s);
    }

    patches.clear();
    patches.resize(specs.size());

    std::vector<char> claimed(n_dofs, 0);
    std::vector<types::global_dof_index> local_dofs(dpc);

    for (unsigned int p = 0; p < specs.size(); ++p)
    {
        const Spec &spec = specs[p];
        Patch &patch = patches[p];

        patch.name = spec.name;

        std::array<unsigned int, 3> n_cells, n_nodes;

        for (unsigned int d = 0; d < 3; ++d)
        {
            n_cells[d] = static_cast<unsigned int>(
                std::lround((spec.hi - spec.lo) / spec.h_patch));

            AssertThrow(n_cells[d] * degree >= 2,
                        ExcMessage("Patch too small."));

            n_nodes[d] = degree * n_cells[d] + 1;
            patch.n_int[d] = n_nodes[d] - 2;

            patch.eig[d] =
                get_eig(degree, n_cells[d], spec.h_patch);
        }

        std::vector<types::global_dof_index> lattice(
            static_cast<std::size_t>(n_nodes[0]) *
                n_nodes[1] * n_nodes[2],
            numbers::invalid_dof_index);

        // Gather the patch's DoF lattice from MatrixFree cell batches.
        for (unsigned int batch = 0; batch < mf.n_cell_batches(); ++batch)
            for (unsigned int v = 0;
                 v < mf.n_active_entries_per_cell_batch(batch); ++v)
            {
                const auto cell = mf.get_cell_iterator(batch, v);

                if (std::abs(
                        cell->extent_in_direction(0) - spec.h_patch) > tol)
                    continue;

                const Point<dim> c = cell->center();

                bool inside = true;

                for (unsigned int d = 0; d < 3; ++d)
                    if (!(c[d] > spec.lo - tol &&
                          c[d] < spec.hi + tol))
                    {
                        inside = false;
                        break;
                    }

                if (!inside)
                    continue;

                std::array<unsigned int, 3> ci;

                for (unsigned int d = 0; d < 3; ++d)
                    ci[d] = static_cast<unsigned int>(
                        std::lround(
                            (c[d] - spec.lo) / spec.h_patch - 0.5));

                cell->get_dof_indices(local_dofs);

                for (unsigned int lz = 0; lz <= degree; ++lz)
                    for (unsigned int ly = 0; ly <= degree; ++ly)
                        for (unsigned int lx = 0; lx <= degree; ++lx)
                        {
                            const unsigned int lex =
                                lx + ly * (degree + 1) +
                                lz * (degree + 1) * (degree + 1);

                            const unsigned int gx =
                                ci[0] * degree + lx;

                            const unsigned int gy =
                                ci[1] * degree + ly;

                            const unsigned int gz =
                                ci[2] * degree + lz;

                            lattice[
                                gx + gy * n_nodes[0] +
                                gz * n_nodes[0] * n_nodes[1]] =
                                local_dofs[l2h[lex]];
                        }
            }

        // Interior, unconstrained, not already claimed.
        for (unsigned int k = 1; k + 1 < n_nodes[2]; ++k)
            for (unsigned int j = 1; j + 1 < n_nodes[1]; ++j)
                for (unsigned int i = 1; i + 1 < n_nodes[0]; ++i)
                {
                    const types::global_dof_index g =
                        lattice[
                            i + j * n_nodes[0] +
                            k * n_nodes[0] * n_nodes[1]];

                    if (g == numbers::invalid_dof_index ||
                        constraints.is_constrained(g) ||
                        claimed[g])
                        continue;

                    claimed[g] = 1;

                    patch.lattice_idx.push_back(
                        (i - 1) +
                        (j - 1) * patch.n_int[0] +
                        (k - 1) * patch.n_int[0] * patch.n_int[1]);

                    patch.local_idx.push_back(
                        partitioner->global_to_local(g));
                }

        const std::size_t total =
            static_cast<std::size_t>(patch.n_int[0]) *
            patch.n_int[1] * patch.n_int[2];

        patch.loc.assign(total, 0.0);
        patch.t1.assign(total, 0.0);
        patch.t2.assign(total, 0.0);

        std::cout
            << "  patch '" << spec.name << "': ["
            << spec.lo << ", " << spec.hi << "]^3, "
            << n_cells[0] << "^3 cells (h=" << spec.h_patch << "), "
            << patch.lattice_idx.size() << " owned DoFs\n";
    }

    // Constrained DoFs -> identity.
    // Uncovered unconstrained DoFs -> Jacobi fallback.
    constrained_local.clear();
    fallback_local.clear();

    for (unsigned int i = 0;
         i < partitioner->locally_owned_size(); ++i)
    {
        const types::global_dof_index g =
            partitioner->local_to_global(i);

        if (constraints.is_constrained(g))
            constrained_local.push_back(i);
        else if (!claimed[g])
            fallback_local.emplace_back(
                i, inverse_diagonal.local_element(i));
    }

    std::cout
        << "  uncovered unconstrained DoFs: "
        << fallback_local.size() << '\n';
}

// ----------------------------------------------------------------------------
// Restricted additive Schwarz application.
//
// Each unconstrained DoF is owned by one patch, so scatter is assignment.
// ----------------------------------------------------------------------------

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::vmult(
    VectorType &dst,
    const VectorType &src) const
{
    double *d = dst.begin();
    const double *s = src.begin();

    // Constrained DoFs: identity.
    for (const unsigned int i : constrained_local)
        d[i] = s[i];

    // Uncovered DoFs: Jacobi fallback.
    for (const auto &fb : fallback_local)
        d[fb.first] = fb.second * s[fb.first];

    // Patch solves.
    for (const Patch &patch : patches)
    {
        std::fill(patch.loc.begin(), patch.loc.end(), 0.0);

        const std::size_t no = patch.lattice_idx.size();

        for (std::size_t k = 0; k < no; ++k)
            patch.loc[patch.lattice_idx[k]] =
                s[patch.local_idx[k]];

        fd_solve(patch);

        for (std::size_t k = 0; k < no; ++k)
            d[patch.local_idx[k]] =
                patch.loc[patch.lattice_idx[k]];
    }
}

// ============================================================================
// Benchmark timing
// ============================================================================

static double wall_seconds()
{
    using Clock = std::chrono::steady_clock;

    return std::chrono::duration<double>(
        Clock::now().time_since_epoch()).count();
}

static void write_benchmark_csv(
    const std::string &path,
    const Parameters &prm,
    const unsigned int fe_degree,
    const unsigned int n_active_cells,
    const types::global_dof_index n_dofs,
    const double setup_time,
    const double t_min,
    const double t_med,
    const double t_p90,
    const double benchmark_wall,
    const double throughput)
{
    std::ifstream check(path);
    const bool has_existing_data =
        check.good() && check.peek() != std::ifstream::traits_type::eof();
    check.close();

    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out)
        throw std::runtime_error("Could not open CSV output: " + path);

    if (!has_existing_data)
    {
        out << "benchmark,fe_degree,n_coarse,active_cells,dofs,overlap,"
               "warmup,reps,setup_time_s,min_vmult_ms,median_vmult_ms,"
               "p90_vmult_ms,avg_vmult_ms,total_wall_s,throughput_MDoF_s\n";
    }

    out << "schwarz_fd," << fe_degree << ',' << prm.n_coarse << ','
        << n_active_cells << ',' << n_dofs << ',' << prm.overlap << ','
        << prm.warmup << ',' << prm.reps << ',' << setup_time << ','
        << t_min * 1e3 << ',' << t_med * 1e3 << ',' << t_p90 * 1e3 << ','
        << (benchmark_wall / prm.reps) * 1e3 << ','
        << benchmark_wall << ',' << throughput << '\n';
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    try
    {
        const Parameters prm = Parameters::parse(argc, argv);

        constexpr int dim = 3;
        constexpr int fe_degree = 4;

        using VectorType =
            LinearAlgebra::distributed::Vector<double>;

        const double domain_start = -10.0;
        const double domain_end   =  10.0;

        // --------------------------------------------------------------------
        // Mesh construction
        // --------------------------------------------------------------------

        Triangulation<dim> tria;

        GridGenerator::subdivided_hyper_cube(
            tria, prm.n_coarse, domain_start, domain_end);

        const double coarse_h =
            (domain_end - domain_start) / prm.n_coarse;

        std::vector<Region> regions;

        regions.push_back({domain_start, domain_end});

        regions.push_back({
            domain_start + prm.mid_first * coarse_h,
            domain_start + (prm.mid_first + prm.mid_count) * coarse_h
        });

        regions.push_back({
            domain_start + prm.inner_first * coarse_h,
            domain_start + (prm.inner_first + prm.inner_count) * coarse_h
        });

        const double tol =
            1e-6 * coarse_h / (1u << (regions.size() - 1));

        for (unsigned int l = 1; l < regions.size(); ++l)
        {
            for (auto &cell : tria.active_cell_iterators())
            {
                const Point<dim> c = cell->center();

                bool inside = true;

                for (unsigned int d = 0; d < dim; ++d)
                    if (!(c[d] > regions[l].lo - tol &&
                          c[d] < regions[l].hi + tol))
                    {
                        inside = false;
                        break;
                    }

                if (inside)
                    cell->set_refine_flag();
            }

            tria.execute_coarsening_and_refinement();
        }

        // --------------------------------------------------------------------
        // DoFs, constraints, MatrixFree operator
        // --------------------------------------------------------------------

        FE_Q<dim> fe(fe_degree);

        DoFHandler<dim> dof_handler(tria);
        dof_handler.distribute_dofs(fe);

        AffineConstraints<double> constraints;

        DoFTools::make_hanging_node_constraints(
            dof_handler, constraints);

        VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            constraints);

        constraints.close();

        MatrixFree<dim, double>::AdditionalData ad;

        ad.mapping_update_flags =
            update_gradients | update_JxW_values;

        auto mf = std::make_shared<MatrixFree<dim, double>>();

        mf->reinit(
            MappingQ1<dim>(),
            dof_handler,
            constraints,
            QGauss<1>(fe_degree + 1),
            ad);

        using LaplaceOp =
            MatrixFreeOperators::LaplaceOperator<
                dim, fe_degree, fe_degree + 1, 1, VectorType>;

        LaplaceOp op;
        op.initialize(mf);
        op.compute_diagonal();

        VectorType inv_diag;

        mf->initialize_dof_vector(inv_diag);

        inv_diag =
            op.get_matrix_diagonal_inverse()->get_vector();

        std::cout
            << "== Setup ==\n"
            << "Polynomial degree: " << fe_degree << '\n'
            << "Coarse cells: " << prm.n_coarse << "^3\n"
            << "Active cells: " << tria.n_active_cells() << '\n'
            << "DoFs: " << dof_handler.n_dofs() << '\n'
            << "Overlap: " << prm.overlap << '\n';

        // --------------------------------------------------------------------
        // Preconditioner setup (excluded from benchmark timings)
        // --------------------------------------------------------------------

        typename TwoFDOverlappingSchwarzPreconditioner<dim>::Geometry geo;

        geo.domain_start = domain_start;
        geo.domain_end = domain_end;
        geo.coarse_h = coarse_h;
        geo.overlap_cells = prm.overlap;
        geo.layers = regions;

        TwoFDOverlappingSchwarzPreconditioner<dim> prec;

        const double setup_start = wall_seconds();

        prec.initialize(*mf, constraints, inv_diag, geo);

        const double setup_time =
            wall_seconds() - setup_start;

        std::cout << "Preconditioner setup: "
                  << setup_time << " s\n";

        // --------------------------------------------------------------------
        // Deterministic input and output vectors
        // --------------------------------------------------------------------

        VectorType src, dst;

        mf->initialize_dof_vector(src);
        mf->initialize_dof_vector(dst);

        std::mt19937 gen(prm.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (unsigned int i = 0;
             i < src.locally_owned_size(); ++i)
            src.local_element(i) = dist(gen);

        // --------------------------------------------------------------------
        // Warm-up
        // --------------------------------------------------------------------

        std::cout
            << "\n== Benchmark ==\n"
            << "Warm-up applications: " << prm.warmup << '\n'
            << "Timed applications: " << prm.reps << '\n';

        for (unsigned int r = 0; r < prm.warmup; ++r)
            prec.vmult(dst, src);

        // --------------------------------------------------------------------
        // Timed vmult calls
        // --------------------------------------------------------------------

        std::vector<double> times;
        times.reserve(prm.reps);

        const double benchmark_start = wall_seconds();

        for (unsigned int r = 0; r < prm.reps; ++r)
        {
            const double t0 = wall_seconds();

            prec.vmult(dst, src);

            const double t1 = wall_seconds();

            times.push_back(t1 - t0);
        }

        const double benchmark_wall =
            wall_seconds() - benchmark_start;

        // --------------------------------------------------------------------
        // Statistics
        // --------------------------------------------------------------------

        std::sort(times.begin(), times.end());

        const double t_min = times.front();
        const double t_med = times[times.size() / 2];

        const std::size_t p90_idx =
            static_cast<std::size_t>(
                0.9 * static_cast<double>(times.size() - 1));

        const double t_p90 = times[p90_idx];

        const double ndofs =
            static_cast<double>(dof_handler.n_dofs());

        std::cout << std::fixed << std::setprecision(6);

        std::cout
            << "\n== vmult performance ==\n"
            << "Minimum:   " << t_min * 1e3 << " ms\n"
            << "Median:    " << t_med * 1e3 << " ms\n"
            << "P90:       " << t_p90 * 1e3 << " ms\n"
            << "Throughput: "
            << ndofs / t_med * 1e-6 << " MDoF/s\n"
            << "Total timed wall time: "
            << benchmark_wall << " s\n"
            << "Average timed call: "
            << benchmark_wall / prm.reps * 1e3 << " ms\n";

        if (!prm.csv_path.empty())
        {
            write_benchmark_csv(
                prm.csv_path,
                prm,
                fe_degree,
                tria.n_active_cells(),
                dof_handler.n_dofs(),
                setup_time,
                t_min,
                t_med,
                t_p90,
                benchmark_wall,
                ndofs / t_med * 1e-6);

            std::cout << "CSV written to: " << prm.csv_path << '\n';
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
