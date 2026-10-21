// Matrix-free Poisson solve  -Laplace(phi) = rho  (eps0 = 1), 3D, locally refined cube, THREE layers.
//   rho : sum of smeared atomic charges (compact support r_c, integrates to the atom's charge)
//   BC  : Dirichlet, phi = sum of point-charge Coulomb potentials q/(4 pi r)
//         (exact outside r_c, so the domain boundary must be farther than r_c from every atom)
//   check: electrostatic energy E = 1/2 int rho*phi, compared with the analytic value
//   layer 0 "outer"  : whole domain, cell size h
//   layer 1 "middle" : middle region, refined once  (h/2)
//   layer 2 "inner"  : inner region, refined twice  (h/4)
//
//   - Jacobi preconditioner (matrix-free diagonal, step-37 style)
//   - Multi-patch fast-diagonalization (FD) Schwarz preconditioner, one patch per layer:
//       inner  patch : inner region  extended by `overlap` middle cells, h/4, acts on inner dofs
//       middle patch : middle region extended by `overlap` coarse cells, h/2, acts on middle dofs (minus inner)
//       outer  patch : entire domain, h, acts on remaining dofs (minus middle and inner)
//     Dof ownership is strictly disjoint (finest layer claims first), so
//     P = sum_p R_p^T A_p^{-1} R_p is symmetric (block-Jacobi with FD blocks) and valid for CG.
//      
//  HANGING NODES -> Identity on constrained dofs
//
// Usage: ./prog [n_coarse] [domain_start] [domain_end]
//               [mid_first] [mid_count] [inner_first] [inner_count] [overlap]
//               [charge] [r_c]
//   all region indices are in COARSE cells, per direction.
//   The inner region must lie at least one coarse cell inside the middle region
//   (otherwise deal.II would add extra refinement to keep the mesh 1-irregular).
//   A single atom sits at the origin (default charge 12, r_c 0.7).
// Defaults: 16 -8 8 3 10 5 6 1 12 0.7   (middle = [-5,5], inner = [-3,3])

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>

using namespace dealii;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
struct Region
{
    double lo = 0.0, hi = 0.0;
};

struct Config
{
    unsigned int n_coarse      = 16;
    double       domain_start  = -8.0;
    double       domain_end    = 8.0;
    unsigned int mid_first     = 3;
    unsigned int mid_count     = 10;
    unsigned int inner_first   = 5;
    unsigned int inner_count   = 6;
    unsigned int overlap_cells = 1;
    double       atom_charge   = 12.0;
    double       atom_rc       = 0.7;

    void validate() const
    {
        AssertThrow(n_coarse >= 1, ExcMessage("n_coarse must be >= 1"));
        AssertThrow(domain_end > domain_start, ExcMessage("domain_end must be > domain_start"));
        AssertThrow(mid_count >= 1 && mid_first + mid_count <= n_coarse,
                    ExcMessage("middle region exceeds the coarse grid"));
        AssertThrow(inner_count >= 1, ExcMessage("inner_count must be >= 1"));
        AssertThrow(inner_first >= mid_first + 1 &&
                        inner_first + inner_count + 1 <= mid_first + mid_count,
                    ExcMessage("inner region must lie at least one coarse cell inside the middle region"));
        AssertThrow(atom_rc > 0.0, ExcMessage("atom_rc must be > 0"));
        // the boundary data q/(4 pi r) is only exact outside the smearing radius
        // (atom sits at the origin)
        AssertThrow(domain_start < -atom_rc && domain_end > atom_rc,
                    ExcMessage("the domain boundary must lie outside the atom's smearing radius"));
    }
};

// ---------------------------------------------------------------------------
// Right-hand side
// ---------------------------------------------------------------------------
template <int dim>
struct Atom
{
    Point<dim> position;
    double     charge = 1.0;
    double     r_c    = 0.7;
};

// Smeared charge density of the atoms, compact support |x - R| <= r_c,
//   rho(r) = q * 21 (r_c - r)^3 (6 r^2 + 3 r r_c + r_c^2) / (5 pi r_c^8),   int rho = q
template <int dim>
class SmearedCharge : public Function<dim>
{
public:
    explicit SmearedCharge(const std::vector<Atom<dim>> &atoms)
        : Function<dim>(1)
        , atoms(atoms)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
        (void)component;

        double total_val = 0.0;

        for (const auto &atom : atoms)
        {
            const double r_c         = atom.r_c;
            const double r_c_squared = r_c * r_c;
            const double r_c_8       = r_c_squared * r_c_squared * r_c_squared * r_c_squared;

            double r_squared = 0.0;
            for (unsigned int i = 0; i < dim; ++i)
            {
                const double d = p[i] - atom.position[i];
                r_squared += d * d;
            }
            const double r = std::sqrt(r_squared);

            if (r <= r_c)
            {
                const double n = -21.0 * (r - r_c) * (r - r_c) * (r - r_c) *
                                 (6.0 * r_squared + 3.0 * r * r_c + r_c_squared);
                const double d = 5.0 * numbers::PI * r_c_8;
                total_val += atom.charge * n / d;
            }
        }

        return total_val;
    }

private:
    const std::vector<Atom<dim>> atoms;
};

// Point-charge Coulomb potential q/(4 pi r); equals the potential of the smeared charge for r >= r_c.
// Used as Dirichlet boundary data.
template <int dim>
class CoulombPotential : public Function<dim>
{
public:
    explicit CoulombPotential(const std::vector<Atom<dim>> &atoms)
        : Function<dim>(1)
        , atoms(atoms)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
        (void)component;

        double total_val = 0.0;

        for (const auto &atom : atoms)
        {
            double r_squared = 0.0;
            for (unsigned int i = 0; i < dim; ++i)
            {
                const double d = p[i] - atom.position[i];
                r_squared += d * d;
            }
            const double r = std::sqrt(r_squared);

            if (r < 1e-12)
                total_val += atom.charge / (4.0 * numbers::PI * 1e-12);
            else
                total_val += atom.charge / (4.0 * numbers::PI * r);
        }

        return total_val;
    }

private:
    const std::vector<Atom<dim>> atoms;
};

// Analytic electrostatic energy E = 1/2 int rho*phi (infinite space) of the smeared charges:
//   self energy of one atom : q^2/(8 pi r_c) * 31924/17875
//   pair interaction        : q_i q_j / (4 pi d_ij)   (valid if the smeared charges do not overlap,
//                                                      d_ij >= r_c,i + r_c,j)
// Returns nothing if any two atoms overlap.
template <int dim>
std::optional<double> analytic_electrostatic_energy(const std::vector<Atom<dim>> &atoms)
{
    double energy = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        energy += atoms[i].charge * atoms[i].charge / (8.0 * numbers::PI * atoms[i].r_c) *
                  (31924.0 / 17875.0);
        for (std::size_t j = i + 1; j < atoms.size(); ++j)
        {
            const double d = atoms[i].position.distance(atoms[j].position);
            if (d < atoms[i].r_c + atoms[j].r_c)
                return std::nullopt;
            energy += atoms[i].charge * atoms[j].charge / (4.0 * numbers::PI * d);
        }
    }
    return energy;
}

// ---------------------------------------------------------------------------
// Matrix-free Laplace operator
// ---------------------------------------------------------------------------
template <int dim, int fe_degree, typename number = double>
class MatrixFreeLaplaceOperator
    : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
    using VectorType = LinearAlgebra::distributed::Vector<number>;

    MatrixFreeLaplaceOperator() = default;

    void clear() override
    {
        MatrixFreeOperators::Base<dim, VectorType>::clear();
    }

    // inverse diagonal as a vector (constrained entries = 1)
    const VectorType &inverse_diagonal() const
    {
        return this->inverse_diagonal_entries->get_vector();
    }

    // Jacobi preconditioner (usable directly in SolverCG)
    const DiagonalMatrix<VectorType> &jacobi() const
    {
        return *this->inverse_diagonal_entries;
    }

    void compute_diagonal() override;

private:
    void apply_add(VectorType &dst, const VectorType &src) const override;

    void local_apply(const MatrixFree<dim, number> &data,
                     VectorType &dst,
                     const VectorType &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_compute_diagonal(const MatrixFree<dim, number> &data,
                                VectorType &dst,
                                const unsigned int &dummy,
                                const std::pair<unsigned int, unsigned int> &cell_range) const;
};

template <int dim, int fe_degree, typename number>
void MatrixFreeLaplaceOperator<dim, fe_degree, number>::local_apply(
    const MatrixFree<dim, number> &data,
    VectorType &dst,
    const VectorType &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
{
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);

        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
    }
}

template <int dim, int fe_degree, typename number>
void MatrixFreeLaplaceOperator<dim, fe_degree, number>::apply_add(VectorType &dst,
                                                                  const VectorType &src) const
{
    this->data->cell_loop(&MatrixFreeLaplaceOperator::local_apply, this, dst, src);
}

template <int dim, int fe_degree, typename number>
void MatrixFreeLaplaceOperator<dim, fe_degree, number>::local_compute_diagonal(
    const MatrixFree<dim, number> &data,
    VectorType &dst,
    const unsigned int &,
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
void MatrixFreeLaplaceOperator<dim, fe_degree, number>::compute_diagonal()
{
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
    VectorType &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    unsigned int dummy = 0;
    this->data->cell_loop(&MatrixFreeLaplaceOperator::local_compute_diagonal,
                          this,
                          inverse_diagonal,
                          dummy);
    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
        const double val = inverse_diagonal.local_element(i);
        inverse_diagonal.local_element(i) = (std::abs(val) > 1e-15) ? 1.0 / val : 1.0;
    }
}

// ---------------------------------------------------------------------------
// Two-patch fast-diagonalization Schwarz preconditioner
// ---------------------------------------------------------------------------
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
        unsigned int overlap_cells;   // patch extension, in cells of the next-coarser layer
        std::vector<Region> layers;   // layers[0] = whole domain, layers[l] refined l times (nested)
    };

    // Cell/dof information is taken from the MatrixFree object (cell batches),
    // so the preconditioner sees exactly the same cells, constraints and vector
    // layout as the matrix-free operator.
    void initialize(const MatrixFree<dim, double> &mf,
                    const AffineConstraints<double> &constraints,
                    const VectorType &inverse_diagonal,
                    const Geometry &geo);

    void vmult(VectorType &dst, const VectorType &src) const;

    double average_apply_time() const
    {
        return n_applications > 0 ? total_apply_time / n_applications : 0.0;
    }

    unsigned int n_uncovered() const { return fallback_local.size(); }

private:
    struct Eig1D
    {
        unsigned int n = 0;
        std::vector<double> S;      // row-major, S[i*n+j]  = j-th eigenvector, component i
        std::vector<double> St;     // row-major, St[j*n+i] = S[i*n+j]
        std::vector<double> lambda;
    };

    struct Patch
    {
        std::string name;
        std::array<unsigned int, 3> n_int{};
        std::array<std::shared_ptr<Eig1D>, 3> eig;
        std::vector<unsigned int> lattice_idx; // position in the patch interior lattice
        std::vector<unsigned int> local_idx;   // local index in the distributed vector
        mutable std::vector<double> loc, t1, t2; // work buffers (no allocation in vmult)
    };

    std::shared_ptr<Eig1D> get_eig(unsigned int degree, unsigned int n_cells, double h);

    static void apply_axis(const std::vector<double> &mat,
                           unsigned int na,
                           const std::vector<double> &in,
                           std::vector<double> &out,
                           const std::array<unsigned int, 3> &n,
                           unsigned int axis);

    void fd_solve(const Patch &patch) const;

    std::map<std::pair<unsigned int, double>, std::shared_ptr<Eig1D>> eig_cache;
    std::vector<Patch> patches;

    std::vector<std::pair<unsigned int, double>> fallback_local; // (local idx, 1/diag) for uncovered dofs
    std::vector<unsigned int> constrained_local;                 // identity on constrained dofs (as in the operator)

    mutable double total_apply_time = 0.0;
    mutable unsigned int n_applications = 0;
};

template <int dim>
std::shared_ptr<typename TwoFDOverlappingSchwarzPreconditioner<dim>::Eig1D>
TwoFDOverlappingSchwarzPreconditioner<dim>::get_eig(unsigned int degree,
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
    const std::vector<unsigned int> l2h = FETools::lexicographic_to_hierarchic_numbering<1>(degree);

    FullMatrix<double> cK(dpc, dpc), cM(dpc, dpc);
    for (unsigned int q = 0; q < quad.size(); ++q)
    {
        const Point<1> pq = quad.point(q);
        const double wq = quad.weight(q);
        for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
            {
                cK(i, j) += fe1.shape_grad(i, pq)[0] * fe1.shape_grad(j, pq)[0] * wq / h;
                cM(i, j) += fe1.shape_value(i, pq) * fe1.shape_value(j, pq) * wq * h;
            }
    }

    FullMatrix<double> K(N, N), M(N, N);
    for (unsigned int e = 0; e < n_cells; ++e)
        for (unsigned int il = 0; il < dpc; ++il)
            for (unsigned int jl = 0; jl < dpc; ++jl)
            {
                K(e * degree + il, e * degree + jl) += cK(l2h[il], l2h[jl]);
                M(e * degree + il, e * degree + jl) += cM(l2h[il], l2h[jl]);
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
    K_int.compute_generalized_eigenvalues_symmetric(M_int,
                                                    -std::numeric_limits<double>::max(),
                                                    std::numeric_limits<double>::max(),
                                                    0.0,
                                                    ev,
                                                    evec,
                                                    1);
    AssertThrow(evec.size() == n, ExcMessage("Unexpected number of eigenvectors"));

    auto e1 = std::make_shared<Eig1D>();
    e1->n = n;
    e1->S.assign(n * n, 0.0);
    e1->St.assign(n * n, 0.0);
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

// Solves (Kx x My x Mz + Mx x Ky x Mz + Mx x My x Kz) u = f in place (u = patch.loc)
template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::fd_solve(const Patch &patch) const
{
    const auto &n = patch.n_int;
    auto &u = patch.loc;
    auto &t1 = patch.t1;
    auto &t2 = patch.t2;

    apply_axis(patch.eig[0]->S, n[0], u, t1, n, 0);
    apply_axis(patch.eig[1]->St, n[1], t1, t2, n, 1);
    apply_axis(patch.eig[2]->St, n[2], t2, t1, n, 2);

    for (unsigned int k = 0; k < n[2]; ++k)
        for (unsigned int j = 0; j < n[1]; ++j)
            for (unsigned int i = 0; i < n[0]; ++i)
                t1[i + j * n[0] + k * n[0] * n[1]] /=
                    (patch.eig[0]->lambda[i] + patch.eig[1]->lambda[j] + patch.eig[2]->lambda[k]);

    apply_axis(patch.eig[2]->S, n[2], t1, t2, n, 2);
    apply_axis(patch.eig[1]->S, n[1], t2, t1, n, 1);
    apply_axis(patch.eig[0]->St, n[0], t1, u, n, 0);
}

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::initialize(const MatrixFree<dim, double> &mf,
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
    AssertThrow(L >= 1, ExcMessage("need at least one layer"));
    const double finest_h = geo.coarse_h / (1u << (L - 1));
    const double tol = 1e-6 * finest_h;

    // lexicographic -> hierarchic map for the 3D cell (dofs of get_dof_indices() are hierarchic)
    const std::vector<unsigned int> l2h = FETools::lexicographic_to_hierarchic_numbering<dim>(degree);
    AssertThrow(l2h.size() == dpc, ExcInternalError());

    struct Spec
    {
        std::string name;
        double lo, hi, h_patch;
    };

    // Finest layer first: it claims its dofs, each coarser layer only gets what is left.
    // Layer l>0: region extended by `overlap` cells of layer l-1 (clamped to the domain).
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
            const double ext = geo.overlap_cells * geo.coarse_h / (1u << (l - 1));
            s.lo = std::max(geo.domain_start, geo.layers[l].lo - ext);
            s.hi = std::min(geo.domain_end, geo.layers[l].hi + ext);
        }
        if (L == 3)
            s.name = (l == 0 ? "outer" : (l == 1 ? "middle" : "inner"));
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
            n_cells[d] = static_cast<unsigned int>(std::lround((spec.hi - spec.lo) / spec.h_patch));
            AssertThrow(n_cells[d] * degree >= 2, ExcMessage("Patch too small."));
            n_nodes[d] = degree * n_cells[d] + 1;
            patch.n_int[d] = n_nodes[d] - 2;
            patch.eig[d] = get_eig(degree, n_cells[d], spec.h_patch);
        }

        std::vector<types::global_dof_index> lattice(
            static_cast<std::size_t>(n_nodes[0]) * n_nodes[1] * n_nodes[2], numbers::invalid_dof_index);

        // Gather lattice from MatrixFree cell batches (only cells of this patch's size)
        for (unsigned int batch = 0; batch < mf.n_cell_batches(); ++batch)
            for (unsigned int v = 0; v < mf.n_active_entries_per_cell_batch(batch); ++v)
            {
                const auto cell = mf.get_cell_iterator(batch, v);

                if (std::abs(cell->extent_in_direction(0) - spec.h_patch) > tol)
                    continue;

                const Point<dim> c = cell->center();
                bool inside = true;
                for (unsigned int d = 0; d < 3; ++d)
                    if (!(c[d] > spec.lo - tol && c[d] < spec.hi + tol))
                    {
                        inside = false;
                        break;
                    }
                if (!inside)
                    continue;

                std::array<unsigned int, 3> ci;
                for (unsigned int d = 0; d < 3; ++d)
                    ci[d] = static_cast<unsigned int>(std::lround((c[d] - spec.lo) / spec.h_patch - 0.5));

                cell->get_dof_indices(local_dofs);
                for (unsigned int lz = 0; lz <= degree; ++lz)
                    for (unsigned int ly = 0; ly <= degree; ++ly)
                        for (unsigned int lx = 0; lx <= degree; ++lx)
                        {
                            const unsigned int lex = lx + ly * (degree + 1) + lz * (degree + 1) * (degree + 1);
                            const unsigned int gx = ci[0] * degree + lx;
                            const unsigned int gy = ci[1] * degree + ly;
                            const unsigned int gz = ci[2] * degree + lz;
                            lattice[gx + gy * n_nodes[0] + gz * n_nodes[0] * n_nodes[1]] =
                                local_dofs[l2h[lex]]; // <-- hierarchic index (fix)
                        }
            }

        // Ownership: interior lattice nodes, unconstrained, not yet claimed
        for (unsigned int k = 1; k + 1 < n_nodes[2]; ++k)
            for (unsigned int j = 1; j + 1 < n_nodes[1]; ++j)
                for (unsigned int i = 1; i + 1 < n_nodes[0]; ++i)
                {
                    const types::global_dof_index g =
                        lattice[i + j * n_nodes[0] + k * n_nodes[0] * n_nodes[1]];
                    if (g == numbers::invalid_dof_index || constraints.is_constrained(g) || claimed[g])
                        continue;

                    claimed[g] = 1;
                    patch.lattice_idx.push_back((i - 1) + (j - 1) * patch.n_int[0] +
                                                (k - 1) * patch.n_int[0] * patch.n_int[1]);
                    patch.local_idx.push_back(partitioner->global_to_local(g));
                }

        const std::size_t tot = static_cast<std::size_t>(patch.n_int[0]) * patch.n_int[1] * patch.n_int[2];
        patch.loc.assign(tot, 0.0);
        patch.t1.assign(tot, 0.0);
        patch.t2.assign(tot, 0.0);

        std::cout << "  patch '" << spec.name << "': [" << spec.lo << ", " << spec.hi << "]^3, "
                  << n_cells[0] << "^3 cells (h=" << spec.h_patch << "), "
                  << patch.lattice_idx.size() << " owned dofs" << std::endl;
    }

    // constrained dofs -> identity (same as MatrixFreeOperators::Base / inverse diagonal = 1)
    // uncovered unconstrained dofs -> Jacobi fallback
    constrained_local.clear();
    fallback_local.clear();
    for (unsigned int i = 0; i < partitioner->locally_owned_size(); ++i)
    {
        const types::global_dof_index g = partitioner->local_to_global(i);
        if (constraints.is_constrained(g))
            constrained_local.push_back(i);
        else if (!claimed[g])
            fallback_local.emplace_back(i, inverse_diagonal.local_element(i));
    }

    std::cout << "  unconstrained dofs with no Schwarz ownership: " << fallback_local.size() << std::endl;
    if (!fallback_local.empty())
        std::cout << "  uncovered dofs use Jacobi fallback instead of Schwarz" << std::endl;
}

template <int dim>
void TwoFDOverlappingSchwarzPreconditioner<dim>::vmult(VectorType &dst, const VectorType &src) const
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    dst = 0;
    for (const auto &patch : patches)
    {
        std::fill(patch.loc.begin(), patch.loc.end(), 0.0);
        const std::size_t n_owned = patch.lattice_idx.size();

        for (std::size_t o = 0; o < n_owned; ++o)
            patch.loc[patch.lattice_idx[o]] = src.local_element(patch.local_idx[o]);

        fd_solve(patch);

        for (std::size_t o = 0; o < n_owned; ++o)
            dst.local_element(patch.local_idx[o]) += patch.loc[patch.lattice_idx[o]];
    }

    for (const auto &f : fallback_local)
        dst.local_element(f.first) += f.second * src.local_element(f.first);

    for (const unsigned int i : constrained_local)
        dst.local_element(i) = src.local_element(i);

    const auto t1 = std::chrono::high_resolution_clock::now();
    total_apply_time += std::chrono::duration<double>(t1 - t0).count();
    ++n_applications;
}

// ---------------------------------------------------------------------------
// Problem
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
class PoissonProblem
{
public:
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    using Operator = MatrixFreeLaplaceOperator<dim, fe_degree, double>;

    explicit PoissonProblem(const std::vector<Atom<dim>> &atoms_in)
        : atoms(atoms_in)
        , fe(fe_degree)
        , dof_handler(triangulation)
    {}

    void setup_system(const Config &cfg);
    void assemble_rhs();
    double compute_electrostatic_energy() const;
    double compute_total_charge() const;

    template <typename PreconditionerType>
    std::pair<unsigned int, double> solve(const PreconditionerType &preconditioner);

    std::vector<Atom<dim>> atoms;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> constraints;
    std::shared_ptr<MatrixFree<dim, double>> mf_storage;
    Operator system_matrix;

    double coarse_h = 0.0;
    std::vector<Region> regions; // [0] = domain, [1] = middle, [2] = inner

    VectorType solution;
    VectorType system_rhs;
};

template <int dim, int fe_degree>
void PoissonProblem<dim, fe_degree>::setup_system(const Config &cfg)
{
    static_assert(dim == 3, "Written for dim == 3.");
    cfg.validate();

    GridGenerator::subdivided_hyper_cube(triangulation, cfg.n_coarse, cfg.domain_start, cfg.domain_end);
    coarse_h = (cfg.domain_end - cfg.domain_start) / cfg.n_coarse;

    regions.clear();
    regions.push_back({cfg.domain_start, cfg.domain_end});
    regions.push_back({cfg.domain_start + cfg.mid_first * coarse_h,
                       cfg.domain_start + (cfg.mid_first + cfg.mid_count) * coarse_h});
    regions.push_back({cfg.domain_start + cfg.inner_first * coarse_h,
                       cfg.domain_start + (cfg.inner_first + cfg.inner_count) * coarse_h});

    const double tol = 1e-6 * coarse_h / (1u << (regions.size() - 1));

    // pass l refines the active cells (level l-1) whose center lies in region l
    for (unsigned int l = 1; l < regions.size(); ++l)
    {
        for (auto &cell : triangulation.active_cell_iterators())
        {
            const Point<dim> c = cell->center();
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
                if (!(c[d] > regions[l].lo - tol && c[d] < regions[l].hi + tol))
                {
                    inside = false;
                    break;
                }
            if (inside)
                cell->set_refine_flag();
        }
        triangulation.execute_coarsening_and_refinement();
    }

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // Dirichlet data: Coulomb potential of the atoms (inhomogeneous constraints)
    VectorTools::interpolate_boundary_values(dof_handler, 0, CoulombPotential<dim>(atoms), constraints);
    constraints.close();

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_gradients | update_JxW_values;

    mf_storage = std::make_shared<MatrixFree<dim, double>>();
    mf_storage->reinit(MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe.degree + 1), additional_data);

    system_matrix.initialize(mf_storage);
    // note: compute_diagonal() is called (and timed) from main()

    mf_storage->initialize_dof_vector(solution);
    mf_storage->initialize_dof_vector(system_rhs);
    solution = 0.0;
    system_rhs = 0.0;

    std::cout << "Domain [" << cfg.domain_start << ", " << cfg.domain_end << "]^3, "
              << cfg.n_coarse << "^3 coarse cells (h=" << coarse_h << "), middle region ["
              << regions[1].lo << ", " << regions[1].hi << "]^3, inner region ["
              << regions[2].lo << ", " << regions[2].hi << "]^3\n"
              << "Active cells: " << triangulation.n_active_cells()
              << ", dofs: " << dof_handler.n_dofs()
              << ", constraints: " << constraints.n_constraints() << std::endl;
}

template <int dim, int fe_degree>
void PoissonProblem<dim, fe_degree>::assemble_rhs()
{
    const SmearedCharge<dim> forcing_function(atoms);

    QGauss<dim> quad(fe.degree + 2);
    FEValues<dim> fe_values(fe, quad,
                            update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> cell_rhs(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    system_rhs = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int q = 0; q < quad.size(); ++q)
        {
            const double f_q = forcing_function.value(fe_values.quadrature_point(q));
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += f_q * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }

        // Inhomogeneous Dirichlet values: the local matrix is needed to move
        // A_ij * g_j to the right-hand side. Only boundary cells need it.
        bool has_inhomogeneity = false;
        for (const auto dof : local_dof_indices)
            if (constraints.is_inhomogeneously_constrained(dof))
            {
                has_inhomogeneity = true;
                break;
            }

        if (has_inhomogeneity)
        {
            cell_matrix = 0.0;
            for (unsigned int q = 0; q < quad.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) +=
                            fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);

            constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs, cell_matrix);
        }
        else
            constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
    }

    system_rhs.compress(VectorOperation::add);
}

// E = 1/2 int rho * phi_h   (call after solve(); `solution` already has the constraints distributed)
template <int dim, int fe_degree>
double PoissonProblem<dim, fe_degree>::compute_electrostatic_energy() const
{
    const SmearedCharge<dim> forcing_function(atoms);
    double total_energy = 0.0;

    const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);

    std::vector<double> solution_values(quadrature_formula.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        const auto &quadrature_points = fe_values.get_quadrature_points();
        fe_values.get_function_values(solution, solution_values); // serial: no ghost update needed

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            const double phi_q = solution_values[q_index];
            const double rho_q = forcing_function.value(quadrature_points[q_index]);
            total_energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index);
        }
    }
    return total_energy;
}

// int rho dx  (should equal the sum of the atom charges; checks that the mesh resolves rho)
template <int dim, int fe_degree>
double PoissonProblem<dim, fe_degree>::compute_total_charge() const
{
    const SmearedCharge<dim> forcing_function(atoms);
    double total = 0.0;

    const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
    FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points | update_JxW_values);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
            total += forcing_function.value(fe_values.quadrature_point(q_index)) * fe_values.JxW(q_index);
    }
    return total;
}

template <int dim, int fe_degree>
template <typename PreconditionerType>
std::pair<unsigned int, double>
PoissonProblem<dim, fe_degree>::solve(const PreconditionerType &preconditioner)
{
    solution = 0.0;

    // relative tolerance 1e-10 on the residual norm
    SolverControl solver_control(3000, 1e-10 * system_rhs.l2_norm());
    solver_control.enable_history_data();
    SolverCG<VectorType> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    const auto &history = solver_control.get_history_data();
    return {solver_control.last_step(), history.empty() ? 0.0 : history.back()};
}

struct TimingRow
{
    std::string method;
    double setup_time = 0.0;
    double solve_time = 0.0;
    unsigned int iterations = 0;
    double residual = 0.0;
    double energy = 0.0;
};

static double seconds(const std::chrono::high_resolution_clock::time_point &a,
                      const std::chrono::high_resolution_clock::time_point &b)
{
    return std::chrono::duration<double>(b - a).count();
}


int main(int argc, char **argv)
{
    constexpr unsigned int fe_degree = 4;
   
    // Parametric sweep: minimum, maximum, and step size.
    const unsigned int n_min  = argc > 1 ? std::stoul(argv[1]) : 24;
    const unsigned int n_max  = argc > 2 ? std::stoul(argv[2]) : 24;
    const unsigned int n_step = argc > 3 ? std::stoul(argv[3]) : 4;

    if (n_step == 0 || n_min > n_max)
        throw std::invalid_argument("Invalid coarse mesh sweep parameters.");

    const std::string csv_path = "schwarz_performance.csv";
    const bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);

    if (!csv)
        throw std::runtime_error("Could not open CSV file: " + csv_path);

    if (!csv_exists)
        csv << "method,n_coarse,mid_count,inner_count,overlap_cells,"
               "active_cells,dofs,setup_time_sec,solve_time_sec,"
               "total_time_sec,cg_iterations,time_per_iter_sec,"
               "final_residual,energy\n";

    // Fixed physical problem across the sweep.
    std::vector<Atom<3>> atoms;
    Atom<3> atom, atom2, atom3, atom4, atom5, atom6;
    
    // Graphene-like configuration: 6 carbon atoms in a hexagon, 1 Angstrom apart.

    atom.position = Point<3>(0, 1, 0);
    atom.charge = 6.0;
    atom.r_c = 0.7;
    atoms.push_back(atom);

    atom2.position = Point<3>(std::sqrt(3) / 2, 0.5, 0);
    atom2.charge = 6.0;
    atom2.r_c = 0.7;
    atoms.push_back(atom2);

    atom3.position = Point<3>(std::sqrt(3) / 2, -0.5, 0);
    atom3.charge = 6.0;
    atom3.r_c = 0.7;
    atoms.push_back(atom3);

    atom4.position = Point<3>(0, -1, 0);
    atom4.charge = 6.0;
    atom4.r_c = 0.7;
    atoms.push_back(atom4);
    
    atom5.position = Point<3>(-std::sqrt(3) / 2, -0.5, 0);
    atom5.charge = 6.0;
    atom5.r_c = 0.7;
    atoms.push_back(atom5);

    atom6.position = Point<3>(-std::sqrt(3) / 2, 0.5, 0);
    atom6.charge = 6.0;
    atom6.r_c = 0.7;
    atoms.push_back(atom6);

    // Refinement configuration for a given coarse mesh.
    const auto make_config = [](const unsigned int n)
    {
        Config cfg;
        cfg.n_coarse = n;

        cfg.mid_first   = n / 4;
        cfg.mid_count   = std::max(1u, n / 2);
        cfg.inner_first = n / 2 - std::max(1u, n / 8);
        cfg.inner_count = std::max(1u, n / 4);

        cfg.overlap_cells = 1;
        return cfg;
    };

    // Record performance and energy for each method.
    const auto record = [&](const std::string &method,
                            const Config &cfg,
                            const PoissonProblem<3, fe_degree> &problem,
                            const double setup_time,
                            const double solve_time,
                            const unsigned int iterations,
                            const double residual,
                            const double energy)
    {
        const double per_iter =
            solve_time / std::max(iterations, 1u);

        csv << std::setprecision(12)
            << method << ','
            << cfg.n_coarse << ','
            << cfg.mid_count << ','
            << cfg.inner_count << ','
            << cfg.overlap_cells << ','
            << problem.triangulation.n_active_cells() << ','
            << problem.dof_handler.n_dofs() << ','
            << setup_time << ','
            << solve_time << ','
            << setup_time + solve_time << ','
            << iterations << ','
            << per_iter << ','
            << residual << ','
            << energy << '\n';

        csv.flush();

        std::cout << "n=" << cfg.n_coarse << " " << method
                  << ": " << iterations << " CG iterations"
                  << ", setup=" << setup_time << " s"
                  << ", solve=" << solve_time << " s"
                  << ", total=" << setup_time + solve_time << " s"
                  << ", DOFs=" << problem.dof_handler.n_dofs()
                  << ", residual=" << residual
                  << ", energy=" << std::setprecision(12) << energy
                  << std::setprecision(6) << '\n';
    };

    // Benchmark sweep.
    for (unsigned int n = n_min; ; )
    {
        const Config cfg = make_config(n);

        PoissonProblem<3, fe_degree> problem(atoms);
        problem.setup_system(cfg);
        problem.assemble_rhs();

        // Compute the diagonal once for Jacobi and Schwarz fallback.
        const auto td0 = std::chrono::high_resolution_clock::now();
        problem.system_matrix.compute_diagonal();
        const auto td1 = std::chrono::high_resolution_clock::now();
        const double diag_time = seconds(td0, td1);

        // ---------------------------------------------------------
        // Jacobi: solve, then calculate energy.
        // ---------------------------------------------------------
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto &jacobi = problem.system_matrix.jacobi();
            const auto t1 = std::chrono::high_resolution_clock::now();

            const auto [iters, res] = problem.solve(jacobi);
            const auto t2 = std::chrono::high_resolution_clock::now();

            const double energy = problem.compute_electrostatic_energy();
            // const double energy = 0.0; // energy computation is commented out for Jacobi

            record("jacobi", cfg, problem,
                   diag_time + seconds(t0, t1),
                   seconds(t1, t2),
                   iters, res, energy);
        }

        // ---------------------------------------------------------
        // Three-layer FD Schwarz: setup, solve, then calculate energy.
        // ---------------------------------------------------------
        {
            TwoFDOverlappingSchwarzPreconditioner<3> schwarz;

            TwoFDOverlappingSchwarzPreconditioner<3>::Geometry geo;
            geo.domain_start = cfg.domain_start;
            geo.domain_end = cfg.domain_end;
            geo.coarse_h = problem.coarse_h;
            geo.overlap_cells = cfg.overlap_cells;
            geo.layers = problem.regions;

            const auto t0 = std::chrono::high_resolution_clock::now();

            schwarz.initialize(*problem.mf_storage,
                               problem.constraints,
                               problem.system_matrix.inverse_diagonal(),
                               geo);

            const auto t1 = std::chrono::high_resolution_clock::now();

            const auto [iters, res] = problem.solve(schwarz);
            const auto t2 = std::chrono::high_resolution_clock::now();

            const double energy = problem.compute_electrostatic_energy();

            const double setup =
                seconds(t0, t1) +
                (schwarz.n_uncovered() > 0 ? diag_time : 0.0);

            record("three_layer_fd_schwarz", cfg, problem,
                   setup,
                   seconds(t1, t2),
                   iters, res, energy);

            std::cout << "  average Schwarz apply time: "
                      << schwarz.average_apply_time() << " s\n";
        }

        // Avoid unsigned overflow when advancing the sweep.
        if (n_max - n < n_step)
            break;

        n += n_step;
    }

    return 0;
}
