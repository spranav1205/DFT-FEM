#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <chrono>
#include <utility>
#include <functional>
#include <array>

using namespace dealii;

// ============================================================================
// 2D validation problem for the FD/Schwarz preconditioner project.
//
// Mesh: an 8x8 uniform coarse grid on [start,end]^2, with the center 2x2
// coarse cells (coarse index (3,3),(3,4),(4,3),(4,4)) refined exactly once,
// so the domain has a single ring of hanging-node interface around a 4x4
// block of fine cells. This is the smallest setup that actually exercises
// the "FD-Schwarz patches must avoid interior hanging nodes" question from
// the 3D uniform-mesh prototype.
//
// RHS / BC: kept deliberately simple (a smooth bump forcing term, zero
// Dirichlet boundary) since the point of this test is the preconditioner
// architecture, not the physics -- unlike the 3D file this does not try to
// reproduce a smeared-charge / Coulomb pair.
// ============================================================================

constexpr double PI = 3.14159265358979323846;

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
        // Smooth compactly-supported-ish bump, sign chosen so the solution
        // is a nice single hump; nothing physical hinges on this.
        return std::exp(-r2 / 4.0);
    }
};

// ============================================================================
// Overlapping (restricted-additive-Schwarz-style) preconditioner, 2D.
//
//   M^{-1} r = sum_i R_i^T W_i A_i^{-1} W_i R_i r
//
// Patches are specified geometrically as axis-aligned bounding boxes
// (xmin, xmax, ymin, ymax) in physical coordinates -- NOT in cell-count
// units -- precisely so a single box can transparently contain a mix of
// coarse and fine cells (the edge and center patches straddle the
// refinement interface).
//
// A patch's interior/owned dofs are exactly those whose support point does
// NOT touch the patch's own bounding box on any side; this is purely a
// geometric trim used for the partition-of-unity overlap (same rule as the
// 3D uniform-mesh version) and has nothing to do with hanging-node status.
//
// The coarse/fine interface line generally falls STRICTLY INSIDE the edge
// and center patch boxes (not on their outer boundary), so those patches do
// contain hanging-node dofs as ordinary interior/owned dofs. Per design,
// this is handled by simply never applying AffineConstraints during patch
// assembly: each patch is assembled by plain per-cell integration over its
// own cells (mixed coarse/fine sizes included), so a touched hanging dof is
// just treated as an ordinary free local unknown, exactly like every other
// dof in the patch.
//
// A_i is the patch's own local stiffness matrix (same bilinear form as the
// global operator), factorized once via UMFPACK. W_i is diagonal, equal to
// 1/(number of patches whose interior contains that dof). Any dof not
// interior to *any* patch (should not happen with the box layout below,
// but guarded against) falls back to a plain Jacobi correction so the
// preconditioner is always well-defined.
// ============================================================================
template <int dim>
class OverlappingSchwarzPreconditioner
{
public:
    struct Box
    {
        std::string name;
        double xmin, xmax, ymin, ymax;
    };

    void initialize(const DoFHandler<dim> &dof_handler,
                     const std::vector<Box> &boxes,
                     const SparseMatrix<double> &system_matrix,
                     double h_fine);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

    double average_apply_time() const
    {
        return n_applications > 0 ? total_apply_time / n_applications : 0.0;
    }

private:
    struct Patch
    {
        std::string name;
        std::vector<types::global_dof_index> local_to_global;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> matrix;
        SparseDirectUMFPACK inverse;
    };

    std::vector<Patch> patches;
    std::vector<double> weight;          // W_i diagonal, indexed by global dof
    std::vector<double> jacobi_fallback; // 1/A_ii, used only where multiplicity == 0

    mutable double total_apply_time = 0.0;
    mutable unsigned int n_applications = 0;
};

template <int dim>
void OverlappingSchwarzPreconditioner<dim>::initialize(
    const DoFHandler<dim> &dof_handler,
    const std::vector<Box> &boxes,
    const SparseMatrix<double> &system_matrix,
    double h_fine)
{
    static_assert(dim == 2, "This preconditioner is written for the 2D validation problem.");

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const double tol = 1e-6 * h_fine;

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

    const unsigned int n_dofs = dof_handler.n_dofs();
    std::vector<unsigned int> multiplicity(n_dofs, 0);

    patches.resize(boxes.size());

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    for (unsigned int p = 0; p < boxes.size(); ++p)
    {
        const Box &box = boxes[p];
        patches[p].name = box.name;

        // Collect every active cell whose center lies inside this box.
        // Because every box boundary in this problem coincides exactly with
        // existing cell boundaries (coarse or fine), a center-point test is
        // unambiguous: a cell is either well inside or well outside.
        std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_cells;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const Point<dim> c = cell->center();
            if (c[0] > box.xmin - tol && c[0] < box.xmax + tol &&
                c[1] > box.ymin - tol && c[1] < box.ymax + tol)
                patch_cells.push_back(cell);
        }

        // Touched dofs, then trim to interior (drop anything on the box
        // boundary -- this is what silently excludes hanging-node dofs,
        // since the refinement interface always sits on a box edge here).
        std::set<types::global_dof_index> touched_dofs;
        for (const auto &cell : patch_cells)
        {
            std::vector<types::global_dof_index> gdi(dofs_per_cell);
            cell->get_dof_indices(gdi);
            for (auto g : gdi)
                touched_dofs.insert(g);
        }

        std::vector<types::global_dof_index> &local_to_global = patches[p].local_to_global;
        std::unordered_map<types::global_dof_index, unsigned int> global_to_local;

        for (auto g : touched_dofs)
        {
            const Point<dim> &sp = support_points.at(g);
            const bool on_boundary =
                std::abs(sp[0] - box.xmin) < tol || std::abs(sp[0] - box.xmax) < tol ||
                std::abs(sp[1] - box.ymin) < tol || std::abs(sp[1] - box.ymax) < tol;
            if (!on_boundary)
            {
                global_to_local[g] = local_to_global.size();
                local_to_global.push_back(g);
                ++multiplicity[g];
            }
        }

        // Assemble A_i directly from the patch's own cells. No
        // AffineConstraints are applied here: any hanging-node dof touched
        // by patch_cells has already been excluded above (it is on the box
        // boundary by construction), so every remaining local dof behaves
        // like an ordinary, unconstrained FE_Q dof for this local problem.
        DynamicSparsityPattern dsp(local_to_global.size(), local_to_global.size());
        std::vector<types::global_dof_index> gdi(dofs_per_cell);

        for (const auto &cell : patch_cells)
        {
            cell->get_dof_indices(gdi);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                auto it_i = global_to_local.find(gdi[i]);
                if (it_i == global_to_local.end()) continue;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    auto it_j = global_to_local.find(gdi[j]);
                    if (it_j == global_to_local.end()) continue;
                    dsp.add(it_i->second, it_j->second);
                }
            }
        }
        patches[p].sparsity_pattern.copy_from(dsp);
        patches[p].matrix.reinit(patches[p].sparsity_pattern);

        QGauss<dim> quad(fe.degree + 1);
        FEValues<dim> fe_values(fe, quad, update_gradients | update_JxW_values);
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

        for (const auto &cell : patch_cells)
        {
            fe_values.reinit(cell);
            cell_matrix = 0;
            for (unsigned int q = 0; q < quad.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) +=
                            fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);

            cell->get_dof_indices(gdi);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                auto it_i = global_to_local.find(gdi[i]);
                if (it_i == global_to_local.end()) continue;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    auto it_j = global_to_local.find(gdi[j]);
                    if (it_j == global_to_local.end()) continue;
                    patches[p].matrix.add(it_i->second, it_j->second, cell_matrix(i, j));
                }
            }
        }

        patches[p].inverse.initialize(patches[p].matrix);

        std::cout << "  Schwarz patch '" << box.name << "': " << patch_cells.size()
                  << " cells, " << local_to_global.size() << " interior dofs" << std::endl;
    }

    weight.assign(n_dofs, 0.0);
    jacobi_fallback.assign(n_dofs, 0.0);
    unsigned int uncovered = 0;
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
        if (multiplicity[i] > 0)
            weight[i] = 1.0 / multiplicity[i];
        else
        {
            ++uncovered;
            const double diag = system_matrix.diag_element(i);
            jacobi_fallback[i] = (diag != 0.0) ? 1.0 / diag : 0.0;
        }
    }
    if (uncovered > 0)
        std::cout << "  NOTE: " << uncovered
                  << " dofs not interior to any Schwarz patch; using Jacobi fallback for them."
                  << std::endl;
    else
        std::cout << "  Every dof (including all hanging-node dofs) is interior to exactly the"
                      " expected number of patches; no fallback needed." << std::endl;
}

template <int dim>
void OverlappingSchwarzPreconditioner<dim>::vmult(Vector<double> &dst, const Vector<double> &src) const
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    dst = 0;
    for (const auto &patch : patches)
    {
        const unsigned int n_local = patch.local_to_global.size();
        Vector<double> r_local(n_local), z_local(n_local);

        for (unsigned int k = 0; k < n_local; ++k)
        {
            const types::global_dof_index g = patch.local_to_global[k];
            r_local[k] = weight[g] * src[g];
        }

        patch.inverse.vmult(z_local, r_local);

        for (unsigned int k = 0; k < n_local; ++k)
        {
            const types::global_dof_index g = patch.local_to_global[k];
            dst[g] += weight[g] * z_local[k];
        }
    }

    // Jacobi fallback for any dof uncovered by every patch.
    for (unsigned int i = 0; i < jacobi_fallback.size(); ++i)
        if (jacobi_fallback[i] != 0.0)
            dst[i] += jacobi_fallback[i] * src[i];

    const auto t1 = std::chrono::high_resolution_clock::now();
    total_apply_time += std::chrono::duration<double>(t1 - t0).count();
    ++n_applications;
}

// ============================================================================
// Poisson problem wrapper
// ============================================================================
template <int dim>
class Poisson
{
public:
    Poisson(int degree);

    void setup_system(unsigned int n_coarse_cells_per_edge, double start, double end);
    void assemble_system(Function<dim> &forcing_function);

    template <typename PreconditionerType>
    std::pair<unsigned int, std::vector<double>>
    solve(bool verbose, const PreconditionerType &preconditioner, const std::string &label);

    void output_results(const std::string &filename, bool verbose) const;

    double coarse_h = 0.0;
    double fine_h = 0.0;

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    PreconditionJacobi<SparseMatrix<double>> jacobi_preconditioner;
    FE_Q<dim> fe;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
};

template <int dim>
Poisson<dim>::Poisson(int degree) : fe(degree), dof_handler(triangulation)
{
}

template <int dim>
void Poisson<dim>::setup_system(unsigned int n_coarse_cells_per_edge, double start, double end)
{
    static_assert(dim == 2, "setup_system's local refinement pattern is written for 2D.");

    GridGenerator::subdivided_hyper_cube(triangulation, n_coarse_cells_per_edge, start, end);
    coarse_h = (end - start) / n_coarse_cells_per_edge;

    // Flag the center 2x2 block of coarse cells (coarse indices 3,4 out of
    // 0..7 for an 8x8 grid) for one refinement, producing a single ring of
    // hanging-node interface around a 4x4 block of fine cells.
    const double lo = start + 3.0 * coarse_h;
    const double hi = start + 5.0 * coarse_h;
    const double tol = 1e-6 * coarse_h;

    for (auto &cell : triangulation.active_cell_iterators())
    {
        const Point<dim> c = cell->center();
        if (c[0] > lo - tol && c[0] < hi + tol && c[1] > lo - tol && c[1] < hi + tol)
            cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement();
    fine_h = coarse_h / 2.0;

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    std::cout << "Number of dofs: " << dof_handler.n_dofs()
              << "  (constraints: " << constraints.n_constraints() << ")" << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
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
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

    jacobi_preconditioner.initialize(system_matrix);
}

template <int dim>
template <typename PreconditionerType>
std::pair<unsigned int, std::vector<double>>
Poisson<dim>::solve(bool verbose, const PreconditionerType &preconditioner, const std::string &label)
{
    solution = 0.0;

    SolverControl solver_control(2000, 1e-10);
    solver_control.enable_history_data();
    SolverCG<Vector<double>> solver(solver_control);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    if (verbose)
        std::cout << "   [" << label << "] " << solver_control.last_step()
                   << " CG iterations needed to obtain convergence." << std::endl;

    return {solver_control.last_step(), solver_control.get_history_data()};
}

template <int dim>
void Poisson<dim>::output_results(const std::string &filename, bool verbose) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "potential");
    data_out.build_patches(fe.degree);

    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (verbose)
        std::cout << "  Results written to " << filename << std::endl;
}

// ============================================================================
// Build the 9 patch boxes: 4 corners, 4 edges, 1 center. All coordinates in
// units of the coarse cell size h, relative to domain start.
// ============================================================================
template <int dim>
std::vector<typename OverlappingSchwarzPreconditioner<dim>::Box>
make_patch_boxes(double start, double h)
{
    using Box = typename OverlappingSchwarzPreconditioner<dim>::Box;
    std::vector<Box> boxes;

    auto X = [&](double units) { return start + units * h; };

    // --- 4 corner patches: 4x4 coarse cells each, pure coarse, no overlap
    //     with the refined block at all. ---
    boxes.push_back({"corner_bl", X(0), X(4), X(0), X(4)});
    boxes.push_back({"corner_br", X(4), X(8), X(0), X(4)});
    boxes.push_back({"corner_tl", X(0), X(4), X(4), X(8)});
    boxes.push_back({"corner_tr", X(4), X(8), X(4), X(8)});

    // --- 4 edge patches: middle two coarse columns/rows, extending 3 coarse
    //     cells in from the domain boundary, then HALF a coarse cell's worth
    //     (one single fine-cell layer, thickness h/2) into the refined
    //     block -- not the full coarse-cell-equivalent (which would be both
    //     stacked fine cells / thickness h). X(3.5) is a genuine mesh line
    //     (the boundary between the two fine cells that make up coarse row
    //     3), so the box edge still lands exactly on a real dof line. ---
    boxes.push_back({"edge_bottom", X(3), X(5), X(0), X(3.5)});
    boxes.push_back({"edge_top", X(3), X(5), X(4.5), X(8)});
    boxes.push_back({"edge_left", X(0), X(3.5), X(3), X(5)});
    boxes.push_back({"edge_right", X(4.5), X(8), X(3), X(5)});

    // --- 1 center patch: ring of coarse cells around the whole 4x4 fine
    //     block (fine-equivalent 8x8 footprint), owning the fine block's
    //     interior dofs. ---
    boxes.push_back({"center", X(2), X(6), X(2), X(6)});

    return boxes;
}

int main()
{
    const int fe_degree = 5;
    const unsigned int n_coarse_cells_per_edge = 8;
    const double domain_start = -8.0, domain_end = 8.0;

    Poisson<2> poisson_problem(fe_degree);
    BumpForcing<2> forcing_term;

    poisson_problem.setup_system(n_coarse_cells_per_edge, domain_start, domain_end);
    poisson_problem.assemble_system(forcing_term);

    std::cout << "\nBuilding Schwarz preconditioner (9 patches: 4 corner + 4 edge + 1 center)..." << std::endl;
    OverlappingSchwarzPreconditioner<2> schwarz;
    {
        const auto boxes = make_patch_boxes<2>(domain_start, poisson_problem.coarse_h);
        const auto t0 = std::chrono::high_resolution_clock::now();
        schwarz.initialize(poisson_problem.dof_handler, boxes, poisson_problem.system_matrix,
                            poisson_problem.fine_h);
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  setup time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    }

    struct Result
    {
        std::string label;
        unsigned int iters;
        double solve_time;
        std::vector<double> history;
    };
    std::vector<Result> results;

    auto run = [&](auto &preconditioner, const std::string &label)
    {
        std::cout << "\n--- Run: " << label << " ---" << std::endl;
        const auto ta = std::chrono::high_resolution_clock::now();
        auto [iters, history] = poisson_problem.solve(true, preconditioner, label);
        const auto tb = std::chrono::high_resolution_clock::now();
        const double t = std::chrono::duration<double>(tb - ta).count();
        results.push_back({label, iters, t, history});
        poisson_problem.output_results("temp2d_" + label + ".vtu", true);
    };

    run(poisson_problem.jacobi_preconditioner, "jacobi");
    run(schwarz, "schwarz9");

    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << std::left
               << std::setw(10) << "label"
               << std::setw(10) << "iters"
               << std::setw(14) << "solve_time_s"
               << std::setw(20) << "s_per_iter_approx"
               << "final_residual" << std::endl;
    for (auto &r : results)
    {
        std::cout << std::left
                   << std::setw(10) << r.label
                   << std::setw(10) << r.iters
                   << std::setw(14) << r.solve_time
                   << std::setw(20) << (r.solve_time / std::max<unsigned int>(r.iters, 1))
                   << (r.history.empty() ? 0.0 : r.history.back()) << std::endl;
    }

    std::cout << "\nSchwarz preconditioner internal average vmult() time:\n";
    std::cout << "  9-patch : " << schwarz.average_apply_time() << " s/application\n";

    std::cout << "\nConvergence history (CG residual per iteration):\n";
    for (auto &r : results)
    {
        std::cout << r.label << ":";
        for (double v : r.history)
            std::cout << " " << v;
        std::cout << std::endl;
    }
}