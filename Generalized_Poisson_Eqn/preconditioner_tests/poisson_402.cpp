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

using namespace dealii;

#define r_c 0.7
#define r_c_squared (r_c * r_c)
#define r_c_8 (r_c_squared * r_c_squared * r_c_squared * r_c_squared)
#define PI 3.14159265358979323846


template <int dim>
class SmearedCharge : public Function<dim>
{
    public:
        SmearedCharge() : Function<dim>() {}
        double charge = 1.0;

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void) component;

            double r_squared = 0;
            for (int i = 0; i < dim; ++i)
                r_squared += p[i] * p[i];

            double r = std::sqrt(r_squared);

            if (r <= r_c)
            {
                double n = -21.0 * (r - r_c) * (r - r_c) * (r - r_c) * (6.0 * r_squared + 3.0 * r * r_c + r_c_squared);
                double d = 5.0 * PI * r_c_8;
                return charge * n / d;
            }
            else
            {
                return 0.0;
            }
        }
};

template <int dim>
class CoullombPotential : public Function<dim>
{
    public:
        CoullombPotential() : Function<dim>() {}
        double charge = 1.0;

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void) component;
            double r_squared = 0;
            for (int i = 0; i < dim; ++i)
                r_squared += p[i] * p[i];

            double r = std::sqrt(r_squared);

            if (r < 1e-12)
                return charge / 1e-12;

            return charge / (4.0 * M_PI * r);
        }
};

// ============================================================================
// Overlapping (restricted-additive-Schwarz-style) preconditioner:
//
//   M^{-1} r = sum_i R_i^T W_i A_i^{-1} W_i R_i r
//
// Subdomains are structured, axis-aligned blocks of cells on the uniform
// mesh, expanded by exactly one cell at every *internal* interface (never
// at the true domain boundary) to produce the requested 1-cell overlap.
//
// R_i is realized implicitly: a patch's local dof set only contains dofs
// that are strictly interior to the patch (support point not touching the
// patch's bounding-box boundary). This is equivalent to assembling the full
// patch operator and then eliminating patch-boundary dofs via homogeneous
// Dirichlet conditions -- i.e. exactly the standard local Schwarz subproblem
// closure -- without ever having to build/zero an explicit Dirichlet row.
//
// A_i is assembled directly from the patch's own cells (same bilinear form
// as the global operator), stored as a small SparseMatrix, and factorized
// once via UMFPACK. This keeps A_i tied to genuine tensor-product structured
// local cell patches, so the direct factorization can later be swapped for
// an FD solve without touching the surrounding preconditioner logic.
//
// W_i is diagonal and depends only on how many patches a given global dof's
// *interior* set contains it in (its overlap multiplicity): weight = 1/mult.
// This is applied symmetrically (before and after the local solve), which
// is what makes sum_i R_i^T W_i R_i = I and keeps the combined operator
// symmetric (important since it feeds CG).
// ============================================================================
template <int dim>
class OverlappingSchwarzPreconditioner
{
    public:
        void initialize(const DoFHandler<dim> &dof_handler,
                         unsigned int n_subdomains,
                         double domain_start,
                         double domain_end,
                         unsigned int n_cells_per_edge);

        void vmult(Vector<double> &dst, const Vector<double> &src) const;

        unsigned int n_patches() const { return patches.size(); }

        double average_apply_time() const
        {
            return n_applications > 0 ? total_apply_time / n_applications : 0.0;
        }

    private:
        struct Patch
        {
            std::vector<types::global_dof_index> local_to_global; // R_i^T index map (interior dofs only)
            SparsityPattern sparsity_pattern;
            SparseMatrix<double> matrix;   // A_i
            SparseDirectUMFPACK inverse;   // factorization of A_i
        };

        std::vector<Patch> patches;
        std::vector<double> weight; // size n_global_dofs, W_i diagonal (same value for every patch containing a dof)

        mutable double total_apply_time = 0.0;
        mutable unsigned int n_applications = 0;
};

template <int dim>
void OverlappingSchwarzPreconditioner<dim>::initialize(
    const DoFHandler<dim> &dof_handler,
    unsigned int n_subdomains,
    double domain_start,
    double domain_end,
    unsigned int n_cells_per_edge)
{
    AssertDimension(dim, 3);
    Assert(n_subdomains == 4 || n_subdomains == 8,
           ExcMessage("Only 4 (1D slab split) or 8 (octant split) subdomains are implemented."));

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const unsigned int N = n_cells_per_edge;
    const double h = (domain_end - domain_start) / N;
    const double tol = 1e-8 * h;

    // Support points for every dof; used purely to classify patch-boundary
    // dofs geometrically (exact for a uniform structured mesh).
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

    // --- Per-patch CORE cell-index ranges (before overlap expansion) ------
    struct Range { unsigned int lo[dim]; unsigned int hi[dim]; };
    std::vector<Range> core_ranges;

    if (n_subdomains == 8)
    {
        // 2 x 2 x 2 octant split.
        for (unsigned int p = 0; p < 8; ++p)
        {
            Range r;
            for (unsigned int d = 0; d < dim; ++d)
            {
                const bool upper_half = (p >> (dim - 1 - d)) & 1u;
                r.lo[d] = upper_half ? N / 2 : 0;
                r.hi[d] = upper_half ? N : N / 2;
            }
            core_ranges.push_back(r);
        }
    }
    else
    {
        // 4 slabs stacked along axis 0; full extent along the other axes.
        for (unsigned int p = 0; p < 4; ++p)
        {
            Range r;
            r.lo[0] = p * N / 4;
            r.hi[0] = (p + 1) * N / 4;
            for (unsigned int d = 1; d < dim; ++d)
            {
                r.lo[d] = 0;
                r.hi[d] = N;
            }
            core_ranges.push_back(r);
        }
    }

    const unsigned int n_dofs = dof_handler.n_dofs();
    std::vector<unsigned int> multiplicity(n_dofs, 0);

    patches.resize(core_ranges.size());

    QGauss<dim> quad(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    for (unsigned int p = 0; p < core_ranges.size(); ++p)
    {
        const Range &core = core_ranges[p];

        // Expand the core by exactly 1 cell at internal interfaces only.
        unsigned int elo[dim], ehi[dim];
        for (unsigned int d = 0; d < dim; ++d)
        {
            elo[d] = (core.lo[d] > 0) ? core.lo[d] - 1 : 0;
            ehi[d] = (core.hi[d] < N) ? core.hi[d] + 1 : N;
        }

        Point<dim> bb_min, bb_max;
        for (unsigned int d = 0; d < dim; ++d)
        {
            bb_min[d] = domain_start + elo[d] * h;
            bb_max[d] = domain_start + ehi[d] * h;
        }

        // Collect the patch's own cells by cell-center index membership.
        std::vector<typename DoFHandler<dim>::active_cell_iterator> patch_cells;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            const Point<dim> c = cell->center();
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
            {
                const unsigned int idx =
                    static_cast<unsigned int>(std::floor((c[d] - domain_start) / h));
                if (idx < elo[d] || idx >= ehi[d])
                {
                    inside = false;
                    break;
                }
            }
            if (inside)
                patch_cells.push_back(cell);
        }

        // Classify dofs touched by the patch: keep interior ones (R_i),
        // drop dofs on the patch's outer (true or artificial) boundary.
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
            bool on_patch_boundary = false;
            for (unsigned int d = 0; d < dim; ++d)
                if (std::abs(sp[d] - bb_min[d]) < tol || std::abs(sp[d] - bb_max[d]) < tol)
                {
                    on_patch_boundary = true;
                    break;
                }
            if (!on_patch_boundary)
            {
                global_to_local[g] = local_to_global.size();
                local_to_global.push_back(g);
                ++multiplicity[g];
            }
        }

        // --- Assemble A_i on interior-interior dof pairs only -------------
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

        std::cout << "  Schwarz patch " << p << ": " << patch_cells.size()
                  << " cells, " << local_to_global.size() << " interior dofs" << std::endl;
    }

    // Partition-of-unity weights: 1 / (number of patches whose interior
    // contains this dof). Handles multiplicity > 2 (e.g. octant corners)
    // automatically.
    weight.assign(n_dofs, 0.0);
    for (unsigned int i = 0; i < n_dofs; ++i)
        if (multiplicity[i] > 0)
            weight[i] = 1.0 / multiplicity[i];

    unsigned int uncovered = 0;
    for (unsigned int i = 0; i < n_dofs; ++i)
        if (multiplicity[i] == 0)
            ++uncovered;
    if (uncovered > 0)
        std::cout << "  WARNING: " << uncovered
                  << " dofs are not interior to any patch (weight 0)." << std::endl;
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
            r_local[k] = weight[g] * src[g];           // W_i R_i r
        }

        patch.inverse.vmult(z_local, r_local);          // A_i^{-1} (W_i R_i r)

        for (unsigned int k = 0; k < n_local; ++k)
        {
            const types::global_dof_index g = patch.local_to_global[k];
            dst[g] += weight[g] * z_local[k];            // R_i^T W_i (...)
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    total_apply_time += std::chrono::duration<double>(t1 - t0).count();
    ++n_applications;
}


template <int dim>
class Poisson
{
    public:
        Poisson(int degree);

        const Vector<double> &get_solution() const;
        const Vector<double> &get_rhs() const;

        void setup_system(int n_cells_per_edge, double start = -1.0, double end = 1.0, CoullombPotential<dim> &boundary_condition = CoullombPotential<dim>());
        void assemble_system(SmearedCharge<dim> &forcing_function);

        template <typename PreconditionerType>
        std::pair<unsigned int, std::vector<double>>
        solve(bool verbose, const PreconditionerType &preconditioner, const std::string &label);

        void output_results(const std::string &filename, bool verbose) const;

        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        PreconditionJacobi<SparseMatrix<double>> jacobi_preconditioner;
        FE_Q<dim> fe;

        AffineConstraints<double> constraints;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double> solution;
        Vector<double> system_rhs;
        Vector<double> system_rhs_before;
};

template <int dim>
Poisson<dim>::Poisson(int degree) : fe(degree), dof_handler(triangulation)
{ }

template <int dim>
const Vector<double>& Poisson<dim>::get_solution() const
{
    return solution;
}

template <int dim>
const Vector<double>& Poisson<dim>::get_rhs() const
{
    return system_rhs_before;
}

template <int dim>
void Poisson<dim>::setup_system(int n_cells_per_edge, double start, double end, CoullombPotential<dim> &boundary_condition)
{
    // Uniform structured mesh only -- no refinement, no hanging nodes.
    GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
    constraints.close();
    std::cout << "Number of constraints: " << constraints.n_constraints() << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    system_rhs_before.reinit(dof_handler.n_dofs());
}

template <int dim>
void Poisson<dim>::assemble_system(SmearedCharge<dim> &forcing_function)
{
    QGauss<dim> matrix_quad(fe.degree + 1);
    QIterated<dim> rhs_quad(QGauss<1>(6), 2);
    FEValues<dim> fe_matrix(fe, matrix_quad, update_values | update_gradients | update_JxW_values);
    FEValues<dim> fe_rhs(fe, rhs_quad, update_values | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_matrix.reinit(cell);
        fe_rhs.reinit(cell);

        const std::vector<Point<dim>> &quadrature_points_r = fe_rhs.get_quadrature_points();

        for (const unsigned int q_index : fe_matrix.quadrature_point_indices())
            for (const unsigned int i : fe_matrix.dof_indices())
                for (const unsigned int j : fe_matrix.dof_indices())
                    cell_matrix(i, j) += fe_matrix.shape_grad(i, q_index) * fe_matrix.shape_grad(j, q_index) * fe_matrix.JxW(q_index);

        for (const unsigned int q_index : fe_rhs.quadrature_point_indices())
        {
            const double f_q = forcing_function.value(quadrature_points_r[q_index]);
            for (const unsigned int i : fe_rhs.dof_indices())
                cell_rhs(i) += f_q * fe_rhs.shape_value(i, q_index) * fe_rhs.JxW(q_index);
        }

        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

    system_rhs_before = system_rhs;

    jacobi_preconditioner.initialize(system_matrix);
}

template <int dim>
template <typename PreconditionerType>
std::pair<unsigned int, std::vector<double>>
Poisson<dim>::solve(bool verbose, const PreconditionerType &preconditioner, const std::string &label)
{
    solution = 0.0; // reset so repeated solves are comparable

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


int main()
{
    // NOTE: degree reduced from 3 (original code) to 2 here so that the
    // dense-ish UMFPACK factorizations of the Schwarz local patches stay
    // cheap at this validation stage. Bump back up once this needs speed.
    const int fe_degree = 3;
    const int n_cells_per_edge = 10;
    const double domain_start = -10.0, domain_end = 10.0;

    Poisson<3> poisson_problem(fe_degree);
    SmearedCharge<3> forcing_term;
    CoullombPotential<3> boundary_term;

    forcing_term.charge = 12.0;
    boundary_term.charge = 12.0;

    poisson_problem.setup_system(n_cells_per_edge, domain_start, domain_end, boundary_term);
    poisson_problem.assemble_system(forcing_term);

    std::cout << "\nBuilding Schwarz preconditioner (4 patches, 1-cell overlap)..." << std::endl;
    OverlappingSchwarzPreconditioner<3> schwarz4;
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        schwarz4.initialize(poisson_problem.dof_handler, 4, domain_start, domain_end, n_cells_per_edge);
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  setup time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    }

    std::cout << "\nBuilding Schwarz preconditioner (8 patches, 1-cell overlap)..." << std::endl;
    OverlappingSchwarzPreconditioner<3> schwarz8;
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        schwarz8.initialize(poisson_problem.dof_handler, 8, domain_start, domain_end, n_cells_per_edge);
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
        poisson_problem.output_results("temp_" + label + ".vtu", true);
    };

    run(poisson_problem.jacobi_preconditioner, "jacobi");
    run(schwarz4, "schwarz4");
    run(schwarz8, "schwarz8");

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

    std::cout << "\nSchwarz preconditioner internal average vmult() time (more precise than the\n"
                  "solve_time/iters proxy above, which also includes CG bookkeeping):\n";
    std::cout << "  4-patch : " << schwarz4.average_apply_time() << " s/application\n";
    std::cout << "  8-patch : " << schwarz8.average_apply_time() << " s/application\n";

    std::cout << "\nConvergence history (CG residual per iteration):\n";
    for (auto &r : results)
    {
        std::cout << r.label << ":";
        for (double v : r.history)
            std::cout << " " << v;
        std::cout << std::endl;
    }
}