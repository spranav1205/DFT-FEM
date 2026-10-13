#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

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
#include <string>
#include <utility>
#include <vector>

using namespace dealii;

// ============================================================================
// Forcing function: Gaussian bump. Centered at the triangle's centroid so it
// actually excites all three refined regions (see main()).
// ============================================================================
template <int dim>
class BumpForcing : public Function<dim>
{
public:
    BumpForcing(const Point<dim> &center = Point<dim>()) : Function<dim>(), c(center) {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double r2 = 0;
        for (unsigned int d = 0; d < dim; ++d)
        {
            const double dd = p[d] - c[d];
            r2 += dd * dd;
        }
        return std::exp(-r2 / 4.0);
    }

private:
    Point<dim> c;
};

// ============================================================================
// Schwarz preconditioner with fast-diagonalization patch solves.
// (unchanged from the original — patch geometry is entirely described by the
// Box list handed to initialize(), so no changes are needed here.)
// ============================================================================
template <int dim>
class FDSchwarzPreconditioner
{
public:
    struct Box
    {
        std::string name;
        std::array<double, dim> lo, hi;         // FD box
        std::array<double, dim> own_lo, own_hi; // region where dofs may be owned
        double h_patch;                         // uniform cell size of the FD lattice
    };

    void initialize(const DoFHandler<dim> &dof_handler,
                    const AffineConstraints<double> &constraints,
                    const std::vector<Box> &boxes,
                    const SparseMatrix<double> &system_matrix);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

    double average_apply_time() const
    {
        return n_applications > 0 ? total_apply_time / n_applications : 0.0;
    }

private:
    struct Eig1D
    {
        unsigned int n = 0;
        FullMatrix<double> S;
        std::vector<double> lambda;
    };

    struct Patch
    {
        std::string name;
        std::array<unsigned int, 3> n_int;
        std::array<std::shared_ptr<Eig1D>, 3> eig;
        std::vector<std::pair<unsigned int, types::global_dof_index>> owned;
    };

    std::shared_ptr<Eig1D> get_eig(unsigned int degree, unsigned int n_cells, double h);

    static void apply_axis(const FullMatrix<double> &A,
                           const std::vector<double> &in,
                           std::vector<double> &out,
                           const std::array<unsigned int, 3> &n,
                           unsigned int axis,
                           bool transpose);

    void fd_solve(const Patch &patch, std::vector<double> &u) const;

    std::map<std::pair<unsigned int, double>, std::shared_ptr<Eig1D>> eig_cache;
    std::vector<Patch> patches;
    std::vector<double> weight;
    std::vector<double> jacobi_fallback;

    mutable double total_apply_time = 0.0;
    mutable unsigned int n_applications = 0;
};

template <int dim>
std::shared_ptr<typename FDSchwarzPreconditioner<dim>::Eig1D>
FDSchwarzPreconditioner<dim>::get_eig(unsigned int degree, unsigned int n_cells, double h)
{
    const auto key = std::make_pair(n_cells, h);
    auto it = eig_cache.find(key);
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
                                                    0.0, ev, evec, 1);
    AssertThrow(evec.size() == n, ExcMessage("Unexpected number of eigenvectors"));

    auto e1 = std::make_shared<Eig1D>();
    e1->n = n;
    e1->S.reinit(n, n);
    e1->lambda.resize(n);
    for (unsigned int j = 0; j < n; ++j)
    {
        e1->lambda[j] = ev[j];
        for (unsigned int i = 0; i < n; ++i)
            e1->S(i, j) = evec[j][i];
    }

    eig_cache[key] = e1;
    return e1;
}

template <int dim>
void FDSchwarzPreconditioner<dim>::apply_axis(const FullMatrix<double> &A,
                                              const std::vector<double> &in,
                                              std::vector<double> &out,
                                              const std::array<unsigned int, 3> &n,
                                              unsigned int axis,
                                              bool transpose)
{
    const unsigned int stride[3] = {1, n[0], n[0] * n[1]};
    const unsigned int na = n[axis];
    const unsigned int sa = stride[axis];

    for (unsigned int k = 0; k < n[2]; ++k)
        for (unsigned int j = 0; j < n[1]; ++j)
            for (unsigned int i = 0; i < n[0]; ++i)
            {
                const unsigned int idx = i + j * n[0] + k * n[0] * n[1];
                const unsigned int ia = (axis == 0) ? i : (axis == 1 ? j : k);
                const unsigned int base = idx - ia * sa;
                double sum = 0.0;
                for (unsigned int m = 0; m < na; ++m)
                    sum += (transpose ? A(m, ia) : A(ia, m)) * in[base + m * sa];
                out[idx] = sum;
            }
}

template <int dim>
void FDSchwarzPreconditioner<dim>::fd_solve(const Patch &patch, std::vector<double> &u) const
{
    const auto &n = patch.n_int;
    const unsigned int tot = n[0] * n[1] * n[2];
    std::vector<double> t1(tot), t2(tot);

    apply_axis(patch.eig[0]->S, u, t1, n, 0, true);
    apply_axis(patch.eig[1]->S, t1, t2, n, 1, true);
    apply_axis(patch.eig[2]->S, t2, t1, n, 2, true);

    for (unsigned int k = 0; k < n[2]; ++k)
        for (unsigned int j = 0; j < n[1]; ++j)
            for (unsigned int i = 0; i < n[0]; ++i)
                t1[i + j * n[0] + k * n[0] * n[1]] /=
                    (patch.eig[0]->lambda[i] + patch.eig[1]->lambda[j] + patch.eig[2]->lambda[k]);

    apply_axis(patch.eig[2]->S, t1, t2, n, 2, false);
    apply_axis(patch.eig[1]->S, t2, t1, n, 1, false);
    apply_axis(patch.eig[0]->S, t1, u, n, 0, false);
}

template <int dim>
void FDSchwarzPreconditioner<dim>::initialize(const DoFHandler<dim> &dof_handler,
                                              const AffineConstraints<double> &constraints,
                                              const std::vector<Box> &boxes,
                                              const SparseMatrix<double> &system_matrix)
{
    static_assert(dim == 3, "Written for dim == 3.");
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const unsigned int degree = fe.degree;
    const unsigned int dpc = fe.n_dofs_per_cell();
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::vector<Point<dim>> support_points(n_dofs);
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

    const std::vector<unsigned int> l2h = FETools::lexicographic_to_hierarchic_numbering<dim>(degree);

    std::vector<unsigned int> multiplicity(n_dofs, 0);
    patches.clear();
    patches.resize(boxes.size());

    std::vector<types::global_dof_index> local_dofs(dpc);

    for (unsigned int p = 0; p < boxes.size(); ++p)
    {
        const Box &box = boxes[p];
        Patch &patch = patches[p];
        patch.name = box.name;
        const double tol = 1e-6 * box.h_patch;

        std::array<unsigned int, 3> n_cells, N;
        for (unsigned int d = 0; d < 3; ++d)
        {
            n_cells[d] = static_cast<unsigned int>(std::lround((box.hi[d] - box.lo[d]) / box.h_patch));
            N[d] = degree * n_cells[d] + 1;
            patch.n_int[d] = N[d] - 2;
            patch.eig[d] = get_eig(degree, n_cells[d], box.h_patch);
        }

        std::vector<types::global_dof_index> lattice(N[0] * N[1] * N[2], numbers::invalid_dof_index);
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (std::abs(cell->extent_in_direction(0) - box.h_patch) > tol)
                continue;
            const Point<dim> c = cell->center();
            bool inside = true;
            for (unsigned int d = 0; d < 3; ++d)
                if (!(c[d] > box.lo[d] - tol && c[d] < box.hi[d] + tol))
                {
                    inside = false;
                    break;
                }
            if (!inside)
                continue;

            std::array<unsigned int, 3> ci;
            for (unsigned int d = 0; d < 3; ++d)
                ci[d] = static_cast<unsigned int>(std::lround((c[d] - box.lo[d]) / box.h_patch - 0.5));

            cell->get_dof_indices(local_dofs);
            for (unsigned int lz = 0; lz <= degree; ++lz)
                for (unsigned int ly = 0; ly <= degree; ++ly)
                    for (unsigned int lx = 0; lx <= degree; ++lx)
                    {
                        const unsigned int lex = lx + ly * (degree + 1) + lz * (degree + 1) * (degree + 1);
                        const unsigned int gx = ci[0] * degree + lx;
                        const unsigned int gy = ci[1] * degree + ly;
                        const unsigned int gz = ci[2] * degree + lz;
                        lattice[gx + gy * N[0] + gz * N[0] * N[1]] = local_dofs[l2h[lex]];
                    }
        }

        for (unsigned int k = 1; k + 1 < N[2]; ++k)
            for (unsigned int j = 1; j + 1 < N[1]; ++j)
                for (unsigned int i = 1; i + 1 < N[0]; ++i)
                {
                    const types::global_dof_index g = lattice[i + j * N[0] + k * N[0] * N[1]];
                    if (g == numbers::invalid_dof_index || constraints.is_constrained(g))
                        continue;
                    const Point<dim> &sp = support_points[g];
                    bool ok = true;
                    for (unsigned int d = 0; d < 3; ++d)
                        if (!(sp[d] > box.own_lo[d] - tol && sp[d] < box.own_hi[d] + tol))
                        {
                            ok = false;
                            break;
                        }
                    if (!ok)
                        continue;
                    patch.owned.emplace_back((i - 1) + (j - 1) * patch.n_int[0] +
                                                 (k - 1) * patch.n_int[0] * patch.n_int[1],
                                             g);
                    ++multiplicity[g];
                }

        std::cout << "  patch '" << box.name << "': lattice " << n_cells[0] << "x" << n_cells[1] << "x"
                  << n_cells[2] << " cells (h=" << box.h_patch << "), " << patch.owned.size()
                  << " owned dofs" << std::endl;
    }

    weight.assign(n_dofs, 0.0);
    jacobi_fallback.assign(n_dofs, 0.0);
    unsigned int uncovered = 0;
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
        if (multiplicity[i] > 0)
            weight[i] = 1.0 / multiplicity[i];
        else if (!constraints.is_constrained(i))
        {
            ++uncovered;
            const double diag = system_matrix.diag_element(i);
            jacobi_fallback[i] = (diag != 0.0) ? 1.0 / diag : 0.0;
        }
    }
    std::cout << "  unconstrained dofs not owned by any patch (Jacobi fallback): " << uncovered << std::endl;
}

template <int dim>
void FDSchwarzPreconditioner<dim>::vmult(Vector<double> &dst, const Vector<double> &src) const
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    dst = 0;
    for (const auto &patch : patches)
    {
        std::vector<double> loc(patch.n_int[0] * patch.n_int[1] * patch.n_int[2], 0.0);
        for (const auto &o : patch.owned)
            loc[o.first] = weight[o.second] * src[o.second];

        fd_solve(patch, loc);

        for (const auto &o : patch.owned)
            dst[o.second] += weight[o.second] * loc[o.first];
    }

    for (unsigned int i = 0; i < jacobi_fallback.size(); ++i)
        if (jacobi_fallback[i] != 0.0)
            dst[i] += jacobi_fallback[i] * src[i];

    const auto t1 = std::chrono::high_resolution_clock::now();
    total_apply_time += std::chrono::duration<double>(t1 - t0).count();
    ++n_applications;
}

// ============================================================================
// Poisson problem
// ============================================================================
template <int dim>
class Poisson
{
public:
    Poisson(int degree) : fe(degree), dof_handler(triangulation) {}

    // Rectangular coarse grid (nx,ny,nz cells of size h, starting at (x0,y0,z0)),
    // with an arbitrary list of axis-aligned refine boxes (each refined once).
    void setup_system(const std::array<unsigned int, 3> &n_cells,
                      double h, double x0, double y0, double z0,
                      const std::vector<std::array<double, 6>> &refine_boxes);
    void assemble_system(Function<dim> &forcing_function);

    template <typename PreconditionerType>
    std::pair<unsigned int, double> solve(const PreconditionerType &preconditioner,
                                        std::vector<double> *residual_history = nullptr);

    double coarse_h = 0.0;
    double fine_h = 0.0;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    PreconditionJacobi<SparseMatrix<double>> jacobi_preconditioner;

    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
};

template <int dim>
void Poisson<dim>::setup_system(const std::array<unsigned int, 3> &n_cells,
                                double h, double x0_, double y0_, double z0_,
                                const std::vector<std::array<double, 6>> &refine_boxes)
{
    x0 = x0_;
    y0 = y0_;
    z0 = z0_;
    coarse_h = h;

    const Point<dim> p1(x0, y0, z0);
    const Point<dim> p2(x0 + n_cells[0] * h, y0 + n_cells[1] * h, z0 + n_cells[2] * h);
    std::vector<unsigned int> repetitions = {n_cells[0], n_cells[1], n_cells[2]};
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

    const double tol = 1e-6 * h;
    for (auto &cell : triangulation.active_cell_iterators())
    {
        const Point<dim> c = cell->center();
        for (const auto &b : refine_boxes)
        {
            if (c[0] > b[0] - tol && c[0] < b[1] + tol &&
                c[1] > b[2] - tol && c[1] < b[3] + tol &&
                c[2] > b[4] - tol && c[2] < b[5] + tol)
            {
                cell->set_refine_flag();
                break;
            }
        }
    }
    triangulation.execute_coarsening_and_refinement();
    fine_h = coarse_h / 2.0;

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    std::cout << "Active cells: " << triangulation.n_active_cells()
              << ", dofs: " << dof_handler.n_dofs()
              << ", constraints: " << constraints.n_constraints() << std::endl;
}

template <int dim>
void Poisson<dim>::assemble_system(Function<dim> &forcing_function)
{
    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fe_values(fe, quad,
                            update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            const double f_q = forcing_function.value(fe_values.quadrature_point(q));
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) +=
                        fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
                cell_rhs(i) += f_q * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                               system_matrix, system_rhs);
    }
}

template <int dim>
template <typename PreconditionerType>
std::pair<unsigned int, double> Poisson<dim>::solve(const PreconditionerType &preconditioner,
                                                      std::vector<double> *residual_history)
{
    solution = 0.0;
    SolverControl solver_control(2000, 1e-10);
    solver_control.enable_history_data();
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    const auto &h = solver_control.get_history_data();
    if (residual_history != nullptr)
        *residual_history = h; // copy (h is a std::vector<double> internally)

    return {solver_control.last_step(), h.empty() ? 0.0 : h.back()};
}
// ============================================================================
// Patch layout: 5 patches.
//
// Coarse grid: nx x ny x nz cells (units of coarse h), refine block placed at
// each triangle vertex (2x2 cells in x,y) spanning the SAME z-band (z cells
// [3,5]) -- i.e. the three refined regions lie in one plane.
//
//   vertexA/B/C : fine FD lattice, box = vertex block +/-1 coarse cell in all
//                 directions, ownership = the vertex's refined 2x2x2 block.
//   top         : full x,y extent, z in [5, nz]      (coarse lattice)
//   bottom      : full x,y extent, z in [0, 3]        (coarse lattice)
//
// Dofs in the mid z-band [3,5] that are NOT inside a vertex box (the gaps
// between triangle vertices) are covered by no patch and fall back to
// Jacobi -- same mechanism the original code already used for uncovered dofs.
// ============================================================================
template <int dim>
std::vector<typename FDSchwarzPreconditioner<dim>::Box>
make_patch_boxes_triangle(double h, double x0, double y0, double z0,
                          unsigned int nx, unsigned int ny, unsigned int nz,
                          const std::array<double, 2> &vA,
                          const std::array<double, 2> &vB,
                          const std::array<double, 2> &vC,
                          double zlo_cells, double zhi_cells)
{
    static_assert(dim == 3, "3D layout.");
    using Box = typename FDSchwarzPreconditioner<dim>::Box;
    std::vector<Box> boxes;
 
    auto Xx = [&](double u) { return x0 + u * h; };
    auto Xy = [&](double u) { return y0 + u * h; };
    auto Xz = [&](double u) { return z0 + u * h; };
 
    auto add_vertex = [&](const std::string &name, const std::array<double, 2> &v) {
        Box b;
        b.name = name;
        b.h_patch = 0.5 * h;
        b.lo = {Xx(v[0] - 1), Xy(v[1] - 1), Xz(zlo_cells - 1)};
        b.hi = {Xx(v[0] + 3), Xy(v[1] + 3), Xz(zhi_cells + 1)};
        b.own_lo = {Xx(v[0]), Xy(v[1]), Xz(zlo_cells)};
        b.own_hi = {Xx(v[0] + 2), Xy(v[1] + 2), Xz(zhi_cells)};
        boxes.push_back(b);
    };
 
    add_vertex("vertexA", vA);
    add_vertex("vertexB", vB);
    add_vertex("vertexC", vC);
 
    // Uniform coarse cuboid, own == box (no padding, matches original end-piece style).
    auto add_rect = [&](const std::string &name, double xlo, double xhi, double ylo, double yhi,
                        double zlo, double zhi) {
        Box b;
        b.name = name;
        b.h_patch = h;
        b.lo = {Xx(xlo), Xy(ylo), Xz(zlo)};
        b.hi = {Xx(xhi), Xy(yhi), Xz(zhi)};
        b.own_lo = b.lo;
        b.own_hi = b.hi;
        boxes.push_back(b);
    };
 
    // --- Wrap the triangle-plane gap (z in [zlo_cells, zhi_cells]) with
    //     rectangular strips carved around the 3 vertex squares. Assumes the
    //     specific layout vA=(2,2), vB=(8,2), vC=(5,8) on a 12x12 (x,y) grid;
    //     adjust the band edges below if vertex placement changes. ---
    add_rect("wrapY0", 0, nx, 0, 2, zlo_cells, zhi_cells);
    add_rect("wrapA_left", 0, 2, 2, 4, zlo_cells, zhi_cells);
    add_rect("wrapA_mid", 4, 8, 2, 4, zlo_cells, zhi_cells);
    add_rect("wrapA_right", 10, nx, 2, 4, zlo_cells, zhi_cells);
    add_rect("wrapY1", 0, nx, 4, 8, zlo_cells, zhi_cells);
    add_rect("wrapC_left", 0, 5, 8, 10, zlo_cells, zhi_cells);
    add_rect("wrapC_right", 7, nx, 8, 10, zlo_cells, zhi_cells);
    add_rect("wrapY2", 0, nx, 10, ny, zlo_cells, zhi_cells);
 
    // top/bottom only need to cover z outside the plane now.
    // add_rect("top", 0, nx, 0, ny, zhi_cells, nz);
    // add_rect("bottom", 0, nx, 0, ny, 0, zlo_cells);
 
    return boxes;
}


int main()
{
    const int fe_degree = 3;
    const double h = 2.0;

    // Coarse grid: 12 x 12 x 8 cells.
    const std::array<unsigned int, 3> n_cells = {12, 12, 8};
    const double x0 = -12.0, y0 = -12.0, z0 = -8.0;

    // Triangle vertices (lower corner of each 2x2 refined block), in coarse
    // cell-index units. Mid z-band: cells [3,5] (same for all three).
    const std::array<double, 2> vA = {2, 2};   // bottom-left
    const std::array<double, 2> vB = {8, 2};   // bottom-right
    const std::array<double, 2> vC = {5, 8};   // top-middle
    const double zlo_cells = 3, zhi_cells = 5;

    std::vector<std::array<double, 6>> refine_boxes;
    auto box_for = [&](const std::array<double, 2> &v) {
        refine_boxes.push_back({x0 + (v[0]) * h, x0 + (v[0] + 2) * h,
                                y0 + (v[1]) * h, y0 + (v[1] + 2) * h,
                                z0 + zlo_cells * h, z0 + zhi_cells * h});
    };
    box_for(vA);
    box_for(vB);
    box_for(vC);

    const std::string csv_path = "schwarz_fd_vs_jacobi_triangle.csv";
    const bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists)
        csv << "method,fe_degree,active_cells,dofs,"
               "setup_time_sec,solve_time_sec,total_time_sec,cg_iterations,"
               "time_per_iter_sec,final_residual\n";

    const std::string history_csv_path = "schwarz_fd_vs_jacobi_triangle_history.csv";
    const bool history_csv_exists = std::filesystem::exists(history_csv_path);
    std::ofstream history_csv(history_csv_path, std::ios::app);
    if (!history_csv_exists)
        history_csv << "method,iteration,residual\n";

    Poisson<3> problem(fe_degree);
    problem.setup_system(n_cells, h, x0, y0, z0, refine_boxes);

    // Centroid of the triangle vertices (in real coords), used to center the bump.
    Point<3> centroid(x0 + (vA[0] + vB[0] + vC[0] + 3.0) / 3.0 * h,
                      y0 + (vA[1] + vB[1] + vC[1] + 3.0) / 3.0 * h,
                      z0 + (zlo_cells + zhi_cells) / 2.0 * h);
    BumpForcing<3> forcing(centroid);
    problem.assemble_system(forcing);

    // Write a row to the CSV file.
    auto write_row = [&](const std::string &method, double setup_t, double solve_t,
                        unsigned int iters, double res, const std::vector<double> &hist) {
        csv << method << "," << fe_degree << "," << problem.triangulation.n_active_cells() << ","
            << problem.dof_handler.n_dofs() << "," << setup_t << "," << solve_t << ","
            << (setup_t + solve_t) << "," << iters << "," << (solve_t / std::max(iters, 1u)) << ","
            << res << std::endl;
        std::cout << method << ": " << iters << " CG iterations, solve " << solve_t
                << " s, " << solve_t / std::max(iters, 1u) << " s/iter, setup " << setup_t
                << " s, final residual " << res << std::endl;

        for (unsigned int it = 0; it < hist.size(); ++it)
            history_csv << method << "," << it << "," << hist[it] << "\n";
    };

    // --- Jacobi ---
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        problem.jacobi_preconditioner.initialize(problem.system_matrix);
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> jac_hist;
        auto [iters, res] = problem.solve(problem.jacobi_preconditioner, &jac_hist);
        const auto t2 = std::chrono::high_resolution_clock::now();
        write_row("jacobi", std::chrono::duration<double>(t1 - t0).count(),
                  std::chrono::duration<double>(t2 - t1).count(), iters, res, jac_hist);
    }

    // --- Schwarz + fast diagonalization: 5 patches (3 vertices + top/bottom) ---
    {
        std::cout << "\nBuilding Schwarz-FD preconditioner (triangle layout)..." << std::endl;
        FDSchwarzPreconditioner<3> fd;
        const auto boxes = make_patch_boxes_triangle<3>(h, x0, y0, z0,
                                                        n_cells[0], n_cells[1], n_cells[2],
                                                        vA, vB, vC, zlo_cells, zhi_cells);
        const auto t0 = std::chrono::high_resolution_clock::now();
        fd.initialize(problem.dof_handler, problem.constraints, boxes, problem.system_matrix);
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> fd_hist;
        auto [iters, res] = problem.solve(fd, &fd_hist);
        const auto t2 = std::chrono::high_resolution_clock::now();
        write_row("schwarz_fd_triangle", std::chrono::duration<double>(t1 - t0).count(),
                  std::chrono::duration<double>(t2 - t1).count(), iters, res, fd_hist);
        std::cout << "  avg vmult time: " << fd.average_apply_time() << " s" << std::endl;
    }

    history_csv.close();
    csv.close();
    return 0;
}