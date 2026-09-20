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
// Geometry knob.
// Extra coarse cells by which the two small end pieces (x in [0,3] / [5,8])
// are extended in y beyond [3,5]. 0 = layout exactly as specified (only the
// z extension). With 0, a thin set of coarse dofs (the y=3 / y=5 lines outside
// the fine block) is interior to no patch and falls back to Jacobi; the code
// prints how many. Setting this to 1 closes that gap.
// ============================================================================
static const double end_piece_y_margin_cells = 0.0;

template <int dim>
class BumpForcing : public Function<dim>
{
public:
    BumpForcing() : Function<dim>() {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double r2 = 0;
        for (unsigned int d = 0; d < dim; ++d)
            r2 += p[d] * p[d];
        return std::exp(-r2 / 4.0);
    }
};

// ============================================================================
// Schwarz preconditioner with fast-diagonalization patch solves.
//
//   M^{-1} r = sum_i R_i^T W_i A_i^{-1} W_i R_i r
//
// Every patch is a box carrying a UNIFORM lattice (all cells the same size,
// either all coarse or all fine) with homogeneous Dirichlet on the box faces.
// A_i^{-1} is applied by fast diagonalization (per-axis generalized
// eigendecompositions of the 1D stiffness/mass matrices).
//
//  * Coarse patches: lattice = real coarse cells. Owned dofs = lattice-interior
//    dofs (unconstrained). Wall dofs (incl. the coarse/fine interface) only
//    act as the Dirichlet wall of the inverse; they are never written.
//  * Center patch: fictitious uniform fine lattice, larger than the refined
//    block. Lattice points that have no real dof (the margin) are buffer.
//    Owned dofs = real unconstrained dofs in the closed refined block,
//    which includes the coarse/fine interface.
//
// Hanging-node / Dirichlet-constrained dofs are never owned. Lattice points
// are located via cell index + local lexicographic numbering (FE_Q uses
// Gauss-Lobatto support points, so positions are not equispaced).
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

    // U = S^T F  (all axes)
    apply_axis(patch.eig[0]->S, u, t1, n, 0, true);
    apply_axis(patch.eig[1]->S, t1, t2, n, 1, true);
    apply_axis(patch.eig[2]->S, t2, t1, n, 2, true);

    for (unsigned int k = 0; k < n[2]; ++k)
        for (unsigned int j = 0; j < n[1]; ++j)
            for (unsigned int i = 0; i < n[0]; ++i)
                t1[i + j * n[0] + k * n[0] * n[1]] /=
                    (patch.eig[0]->lambda[i] + patch.eig[1]->lambda[j] + patch.eig[2]->lambda[k]);

    // u = S U
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

        // Map real dofs onto the (uniform) patch lattice via cell index + local lex numbering.
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

        // Owned dofs: lattice-interior, real, unconstrained, inside the ownership region.
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

    void setup_system(unsigned int n_coarse_cells_per_edge, double start, double end);
    void assemble_system(Function<dim> &forcing_function);

    template <typename PreconditionerType>
    std::pair<unsigned int, double> solve(const PreconditionerType &preconditioner);

    double coarse_h = 0.0;
    double fine_h = 0.0;

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
void Poisson<dim>::setup_system(unsigned int n_coarse_cells_per_edge, double start, double end)
{
    GridGenerator::subdivided_hyper_cube(triangulation, n_coarse_cells_per_edge, start, end);
    coarse_h = (end - start) / n_coarse_cells_per_edge;

    // Refine the center 2x2x2 coarse block (indices 3,4 of 0..7) once.
    const double lo = start + 3.0 * coarse_h;
    const double hi = start + 5.0 * coarse_h;
    const double tol = 1e-6 * coarse_h;
    for (auto &cell : triangulation.active_cell_iterators())
    {
        const Point<dim> c = cell->center();
        bool inside = true;
        for (unsigned int d = 0; d < dim; ++d)
            if (!(c[d] > lo - tol && c[d] < hi + tol))
            {
                inside = false;
                break;
            }
        if (inside)
            cell->set_refine_flag();
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
std::pair<unsigned int, double> Poisson<dim>::solve(const PreconditionerType &preconditioner)
{
    solution = 0.0;
    SolverControl solver_control(2000, 1e-10);
    solver_control.enable_history_data();
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    const auto &h = solver_control.get_history_data();
    return {solver_control.last_step(), h.empty() ? 0.0 : h.back()};
}

// ============================================================================
// Patch layout (units of coarse h on the 8h cube; refined block = [3,5]^3):
//   center      : fine FD lattice [2,6]^3, owns dofs in closed [3,5]^3
//   top/bottom  : x,y in [0,8], z in [0,3] / [5,8]
//   long sides  : x in [0,8], y in [0,3] / [5,8], z in [2,6]
//   small ends  : x in [0,3] / [5,8], y in [3,5], z in [2,6]
// All non-center patches are uniform coarse lattices ending at (never inside)
// the fine block.
// ============================================================================
template <int dim>
std::vector<typename FDSchwarzPreconditioner<dim>::Box>
make_patch_boxes(double start, double h)
{
    static_assert(dim == 3, "3D layout.");
    using Box = typename FDSchwarzPreconditioner<dim>::Box;
    std::vector<Box> boxes;
    auto X = [&](double u) { return start + u * h; };

    auto add = [&](const std::string &name, std::array<double, 2> x, std::array<double, 2> y,
                   std::array<double, 2> z, double hp) {
        Box b;
        b.name = name;
        b.h_patch = hp;
        b.lo = {X(x[0]), X(y[0]), X(z[0])};
        b.hi = {X(x[1]), X(y[1]), X(z[1])};
        b.own_lo = b.lo;
        b.own_hi = b.hi;
        boxes.push_back(b);
        return boxes.size() - 1;
    };

    // Center (fine): FD box [2,6]^3, ownership region = refined block [3,5]^3
    {
        const auto i = add("center", {2, 6}, {2, 6}, {2, 6}, 0.5 * h);
        for (unsigned int d = 0; d < 3; ++d)
        {
            boxes[i].own_lo[d] = X(3);
            boxes[i].own_hi[d] = X(5);
        }
    }

    add("top", {0, 8}, {0, 8}, {5, 8}, h);
    add("bottom", {0, 8}, {0, 8}, {0, 3}, h);

    add("side_y_lo", {0, 8}, {0, 3}, {2, 6}, h);
    add("side_y_hi", {0, 8}, {5, 8}, {2, 6}, h);

    const double m = end_piece_y_margin_cells;
    add("end_x_lo", {0, 3}, {3 - m, 5 + m}, {2, 6}, h);
    add("end_x_hi", {5, 8}, {3 - m, 5 + m}, {2, 6}, h);

    return boxes;
}

int main()
{
    const int fe_degree = 3;
    const unsigned int n_coarse_cells_per_edge = 8;
    const double domain_start = -8.0, domain_end = 8.0;

    const std::string csv_path = "schwarz_fd_vs_jacobi.csv";
    const bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv_exists)
        csv << "method,fe_degree,coarse_cells_per_edge,active_cells,dofs,"
               "setup_time_sec,solve_time_sec,total_time_sec,cg_iterations,"
               "time_per_iter_sec,final_residual\n";

    Poisson<3> problem(fe_degree);
    BumpForcing<3> forcing;
    problem.setup_system(n_coarse_cells_per_edge, domain_start, domain_end);
    problem.assemble_system(forcing);

    auto write_row = [&](const std::string &method, double setup_t, double solve_t,
                         unsigned int iters, double res) {
        csv << method << "," << fe_degree << "," << n_coarse_cells_per_edge << ","
            << problem.triangulation.n_active_cells() << "," << problem.dof_handler.n_dofs() << ","
            << setup_t << "," << solve_t << "," << (setup_t + solve_t) << "," << iters << ","
            << (solve_t / std::max(iters, 1u)) << "," << res << std::endl;
        std::cout << method << ": " << iters << " CG iterations, solve " << solve_t
                  << " s, " << solve_t / std::max(iters, 1u) << " s/iter, setup " << setup_t
                  << " s, final residual " << res << std::endl;
    };

    // --- Jacobi ---
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        problem.jacobi_preconditioner.initialize(problem.system_matrix);
        const auto t1 = std::chrono::high_resolution_clock::now();
        auto [iters, res] = problem.solve(problem.jacobi_preconditioner);
        const auto t2 = std::chrono::high_resolution_clock::now();
        write_row("jacobi", std::chrono::duration<double>(t1 - t0).count(),
                  std::chrono::duration<double>(t2 - t1).count(), iters, res);
    }

    // --- Schwarz + fast diagonalization ---
    {
        std::cout << "\nBuilding Schwarz-FD preconditioner..." << std::endl;
        FDSchwarzPreconditioner<3> fd;
        const auto boxes = make_patch_boxes<3>(domain_start, problem.coarse_h);
        const auto t0 = std::chrono::high_resolution_clock::now();
        fd.initialize(problem.dof_handler, problem.constraints, boxes, problem.system_matrix);
        const auto t1 = std::chrono::high_resolution_clock::now();
        auto [iters, res] = problem.solve(fd);
        const auto t2 = std::chrono::high_resolution_clock::now();
        write_row("schwarz_fd", std::chrono::duration<double>(t1 - t0).count(),
                  std::chrono::duration<double>(t2 - t1).count(), iters, res);
        std::cout << "  avg vmult time: " << fd.average_apply_time() << " s" << std::endl;
    }

    csv.close();
    return 0;
}