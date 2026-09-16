#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>   // SparseDirectUMFPACK for P^{-1}
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>

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
// First-order Neumann (additive Schwarz-style) preconditioner:
//   M^{-1} = P^{-1} - P^{-1} Q P^{-1}
// applied as: u = P^{-1} r ; v = Q u ; w = P^{-1} v ; z = u - w
//
// P^{-1} is realized via a direct (UMFPACK) factorization of the assembled
// P matrix. This is deliberately exact/expensive for now -- the point of
// this version is only to check whether the P/Q split idea reduces CG
// iterations at all, before replacing P^{-1} with an FD solve.
// ============================================================================
class NeumannPQPreconditioner
{
    public:
        void initialize(const SparseMatrix<double> &P_in, const SparseMatrix<double> &Q_in)
        {
            P = &P_in;
            Q = &Q_in;
            P_inv.initialize(*P);
        }

        void vmult(Vector<double> &dst, const Vector<double> &src) const
        {
            Vector<double> u(src.size());
            Vector<double> v(src.size());
            Vector<double> w(src.size());
            Vector<double> t(src.size());

            u = src;
            P_inv.vmult(u, src);      // u = P^{-1} r

            Q->vmult(v, u);           // v = Q P^{-1} r
            P_inv.vmult(w, v);        // w = P^{-1} Q P^{-1} r

            Q->vmult(v, w);           // v = Q P^{-1} Q P^{-1} r
            P_inv.vmult(t, v);        // t = P^{-1} Q P^{-1} Q P^{-1} r

            dst = u;
            dst -= w;
            dst += t;                 // z = u - w + t
        }

    private:
        SmartPointer<const SparseMatrix<double>> P;
        SmartPointer<const SparseMatrix<double>> Q;
        SparseDirectUMFPACK P_inv;
};


template <int dim>
class Poisson
{
    public:
        Poisson(int degree);

        const Vector<double> &get_solution() const;
        const Vector<double> &get_rhs() const;

        void setup_system(int n_cells_per_edge, double start = -1.0, double end = 1.0, CoullombPotential<dim> &boundary_condition = CoullombPotential<dim>());
        void assemble_system(SmearedCharge<dim> &forcing_function);
        void verify_PQ_split(bool verbose) const;
        int solve(bool verbose, bool use_custom_preconditioner);
        void output_results(const std::string &filename, bool verbose) const;

        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        PreconditionJacobi<SparseMatrix<double>> jacobi_preconditioner;
        NeumannPQPreconditioner pq_preconditioner;
        FE_Q<dim> fe;

        AffineConstraints<double> constraints;

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        SparseMatrix<double> P_matrix; // coarse (unrefined, level == 0) cell contributions
        SparseMatrix<double> Q_matrix; // fine (refined, level != 0) cell contributions
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
    int marked = 0;
    GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, start, end);
    for (unsigned int cycle = 0; cycle < 1; ++cycle)
    {
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if (cell->center().norm() < 4.0)
            {
                cell->set_refine_flag();
                marked++;
            }
        }
        std::cout << "Marked " << marked << " cells\n";
        triangulation.execute_coarsening_and_refinement();
    }

    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_condition, constraints);
    constraints.close();
    std::cout << "Number of constraints: " << constraints.n_constraints() << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    // P and Q share the same (superset) sparsity pattern as the full system.
    // Each individually only populates the subset of entries touched by its
    // own cells; the rest of the pattern's entries stay exactly zero.
    P_matrix.reinit(sparsity_pattern);
    Q_matrix.reinit(sparsity_pattern);

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

        // Full system matrix + rhs, as before.
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

        // P/Q split: every active cell is either level 0 (never refined --
        // "coarse") or level != 0 (a child of a refined cell -- "fine").
        // Each cell's local matrix goes entirely into exactly one of P, Q,
        // so summing the two assembled matrices reproduces A exactly.
        if (cell->level() == 0)
            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, P_matrix);
        else
            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, Q_matrix);
    }

    system_rhs_before = system_rhs;

    jacobi_preconditioner.initialize(system_matrix);

    // --- Regularize P ---------------------------------------------------
    // DOFs strictly interior to a refined patch are only ever touched by
    // fine (Q) cells, so their row in P is identically zero. A direct
    // factorization of P as-is would be singular. For this correctness
    // check we patch those rows with a unit diagonal so P is invertible;
    // this is a stand-in and will need a proper fix once P^{-1} is
    // replaced by FD.
    unsigned int n_patched = 0;
    for (unsigned int i = 0; i < P_matrix.m(); ++i)
    {
        if (P_matrix.diag_element(i) == 0.0)
        {
            Q_matrix.set(i, i, Q_matrix.diag_element(i) - 1.0);
            P_matrix.set(i, i, 1.0);
            ++n_patched;
        }
    }
    std::cout << "P matrix: patched " << n_patched << " zero-diagonal rows (fine-only interior dofs)." << std::endl;

    pq_preconditioner.initialize(P_matrix, Q_matrix);

    verify_PQ_split(true);
}

template <int dim>
void Poisson<dim>::verify_PQ_split(bool verbose) const
{
    // Check A x ~= P x + Q x for a handful of random test vectors.
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const unsigned int n = dof_handler.n_dofs();
    double max_rel_error = 0.0;

    for (unsigned int trial = 0; trial < 5; ++trial)
    {
        Vector<double> x(n);
        for (unsigned int i = 0; i < n; ++i)
            x(i) = dist(gen);

        Vector<double> Ax(n), Px(n), Qx(n), sum(n), diff(n);
        system_matrix.vmult(Ax, x);
        P_matrix.vmult(Px, x);
        Q_matrix.vmult(Qx, x);
        sum = Px;
        sum += Qx;
        diff = Ax;
        diff -= sum;

        const double rel_error = diff.l2_norm() / (Ax.l2_norm() > 1e-14 ? Ax.l2_norm() : 1.0);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    if (verbose)
        std::cout << "P/Q split check: max relative ||Ax - (Px+Qx)|| over 5 random vectors = "
                   << std::scientific << max_rel_error << std::defaultfloat << std::endl;
}

template <int dim>
int Poisson<dim>::solve(bool verbose, bool use_custom_preconditioner)
{
    solution = 0.0; // reset so repeated solves are comparable

    SolverControl solver_control(2000, 1e-10);
    SolverCG<Vector<double>> solver(solver_control);

    if (use_custom_preconditioner)
        solver.solve(system_matrix, solution, system_rhs, pq_preconditioner);
    else
        solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);

    constraints.distribute(solution);

    if (verbose)
    {
        std::cout << "   [" << (use_custom_preconditioner ? "Neumann P/Q" : "Jacobi") << "] "
                  << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }

    return solver_control.last_step();
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
    Poisson<3> poisson_problem(3);
    SmearedCharge<3> forcing_term;
    CoullombPotential<3> boundary_term;

    forcing_term.charge = 12.0;
    boundary_term.charge = 12.0;

    poisson_problem.setup_system(8, -10.0, 10.0, boundary_term);
    poisson_problem.assemble_system(forcing_term); // also builds P, Q and verifies the split

    std::cout << "\n--- Run 1: Jacobi preconditioner ---" << std::endl;
    int jacobi_steps = poisson_problem.solve(true, /*use_custom_preconditioner=*/false);
    poisson_problem.output_results("temp_jacobi.vtu", true);

    std::cout << "\n--- Run 2: Neumann P/Q preconditioner ---" << std::endl;
    int pq_steps = poisson_problem.solve(true, /*use_custom_preconditioner=*/true);
    poisson_problem.output_results("temp_pq.vtu", true);

    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Jacobi CG iterations     : " << jacobi_steps << std::endl;
    std::cout << "Neumann P/Q CG iterations: " << pq_steps << std::endl;
}