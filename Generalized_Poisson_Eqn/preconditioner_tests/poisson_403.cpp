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
#include <string>

using namespace dealii;

// ============================================================================
// 3D validation problem, step 2: single locally-refined center block, but now
// with the full corner/face/center patch family (this is the geometry we'll
// reuse per-block once we move to multiple scattered refined regions).
//
// Mesh: an 8x8x8 uniform coarse grid on [start,end]^3, with the center 2x2x2
// coarse block (coarse index {3,4} in every axis) refined exactly once,
// giving one ring of hanging-node interface around a 4x4x4 fine block.
// ============================================================================

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
// Overlapping (restricted-additive-Schwarz-style) preconditioner.
//
//   M^{-1} r = sum_i R_i^T W_i A_i^{-1} W_i R_i r
//
// Patches are axis-aligned bounding boxes in PHYSICAL coordinates (not cell
// counts), so a box transparently contains a mix of coarse and fine cells.
//
// A patch's interior/owned dofs are exactly those whose support point does
// NOT touch the patch's own box on any side -- purely geometric, and used
// only for the partition-of-unity overlap weighting. The coarse/fine
// interface generally falls STRICTLY INSIDE the corner, face, and center
// patch boxes (not on their outer boundary), so those patches do contain
// hanging-node dofs as ordinary interior/owned dofs. This is handled by
// simply never applying AffineConstraints during patch assembly: each patch
// is assembled by plain per-cell integration over its own cells (mixed
// coarse/fine sizes included), so a touched hanging dof is just treated as
// an ordinary free local unknown, like every other dof in the patch.
//
// A_i is the patch's own local stiffness matrix, factorized once via
// UMFPACK. W_i is diagonal, equal to 1/(number of patches whose interior
// contains that dof). Any dof not interior to any patch falls back to a
// Jacobi correction so the preconditioner is always well-defined.
// ============================================================================
template <int dim>
class OverlappingSchwarzPreconditioner
{
public:
    struct Box
    {
        std::string name;
        std::array<double, dim> lo, hi;
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
    std::vector<double> weight;
    std::vector<double> jacobi_fallback;

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

        std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_cells;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const Point<dim> c = cell->center();
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
                if (!(c[d] > box.lo[d] - tol && c[d] < box.hi[d] + tol))
                {
                    inside = false;
                    break;
                }
            if (inside)
                patch_cells.push_back(cell);
        }

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
            bool on_boundary = false;
            for (unsigned int d = 0; d < dim; ++d)
                if (std::abs(sp[d] - box.lo[d]) < tol || std::abs(sp[d] - box.hi[d]) < tol)
                {
                    on_boundary = true;
                    break;
                }
            if (!on_boundary)
            {
                global_to_local[g] = local_to_global.size();
                local_to_global.push_back(g);
                ++multiplicity[g];
            }
        }

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
        std::cout << "  Every dof (including all hanging-node dofs) is interior to at least one"
                     " patch; no fallback needed." << std::endl;
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
    GridGenerator::subdivided_hyper_cube(triangulation, n_coarse_cells_per_edge, start, end);
    coarse_h = (end - start) / n_coarse_cells_per_edge;

    // Flag the center 2x2x2 block of coarse cells (coarse indices 3,4 out of
    // 0..7 for an 8x8x8 grid) for one refinement -> a single ring of
    // hanging-node interface around a 4x4x4 block of fine cells.
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
// Build the 15 patch boxes: 8 corner (octant), 6 face, 1 center, all around
// a single refined 2x2x2 coarse block at indices {3,4} in every axis of an
// 8x8x8 coarse grid. Coordinates in units of the coarse cell size h.
//
//   - corners: full octants [0,4) or [4,8) per axis (same split as the
//     original uniform-mesh 3D code). These naturally eat the innermost
//     refined coarse cell on their corner; the coarse/fine interface ends
//     up strictly interior to the patch, which is fine (see class comment).
//   - faces: narrow (2 coarse cells, indices {3,4}) in the two axes
//     parallel to that face, bleeding from the domain boundary inward along
//     the third axis only to the HALF-cell mark (one single fine-cell
//     layer, not the full innermost coarse cell -- that's already claimed
//     by the corners). X(3.5)/X(4.5) are genuine fine-fine mesh lines.
//   - center: one coarse-cell margin around the whole refined block.
// ============================================================================
template <int dim>
std::vector<typename OverlappingSchwarzPreconditioner<dim>::Box>
make_patch_boxes(double start, double h)
{
    static_assert(dim == 3, "This layout is written for the single-center-block 3D case.");
    using Box = typename OverlappingSchwarzPreconditioner<dim>::Box;
    std::vector<Box> boxes;

    auto X = [&](double units) { return start + units * h; };

    // --- 8 corner (octant) patches ---
    for (unsigned int p = 0; p < 8; ++p)
    {
        Box b;
        b.name = "corner_" + std::to_string(p);
        for (unsigned int d = 0; d < dim; ++d)
        {
            const bool upper = (p >> d) & 1u;
            b.lo[d] = upper ? X(4) : X(0);
            b.hi[d] = upper ? X(8) : X(4);
        }
        boxes.push_back(b);
    }

    // --- 6 face patches ---
    const char *axis_name[3] = {"x", "y", "z"};
    for (unsigned int axis = 0; axis < dim; ++axis)
    {
        for (int side = 0; side < 2; ++side)
        {
            Box b;
            b.name = std::string("face_") + axis_name[axis] + (side == 0 ? "_lo" : "_hi");
            for (unsigned int d = 0; d < dim; ++d)
            {
                if (d == axis)
                {
                    b.lo[d] = (side == 0) ? X(0) : X(4.5);
                    b.hi[d] = (side == 0) ? X(3.5) : X(8);
                }
                else
                {
                    b.lo[d] = X(3);
                    b.hi[d] = X(5);
                }
            }
            boxes.push_back(b);
        }
    }

    // --- 1 center patch, bigger: 1-coarse-cell margin around the 2x2x2
    //     refined block ---
    {
        Box b;
        b.name = "center";
        for (unsigned int d = 0; d < dim; ++d)
        {
            b.lo[d] = X(2);
            b.hi[d] = X(6);
        }
        boxes.push_back(b);
    }

    return boxes;
}

int main()
{
    const int fe_degree = 3;
    const unsigned int n_coarse_cells_per_edge = 8;
    const double domain_start = -8.0, domain_end = 8.0;

    Poisson<3> poisson_problem(fe_degree);
    BumpForcing<3> forcing_term;

    poisson_problem.setup_system(n_coarse_cells_per_edge, domain_start, domain_end);
    poisson_problem.assemble_system(forcing_term);

    std::cout << "\nBuilding Schwarz preconditioner (8 corner + 6 face + 1 center = 15 patches)..."
              << std::endl;
    OverlappingSchwarzPreconditioner<3> schwarz;
    {
        const auto boxes = make_patch_boxes<3>(domain_start, poisson_problem.coarse_h);
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
        poisson_problem.output_results("temp3d_" + label + ".vtu", true);
    };

    run(poisson_problem.jacobi_preconditioner, "jacobi");
    run(schwarz, "schwarz15");

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
    std::cout << "  15-patch : " << schwarz.average_apply_time() << " s/application\n";

    std::cout << "\nConvergence history (CG residual per iteration):\n";
    for (auto &r : results)
    {
        std::cout << r.label << ":";
        for (double v : r.history)
            std::cout << " " << v;
        std::cout << std::endl;
    }
}