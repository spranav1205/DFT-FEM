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
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/fe/fe_tools.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace FastDiagonalization
{
    using namespace dealii;

    template <int dim>
    struct Atom
    {
        Point<dim> position;
        double     charge = 1.0;
        double     r_c    = 0.7;
    };

    template <int dim>
    class SmearedCharge : public Function<dim>
    {
    public:
        explicit SmearedCharge(const std::vector<Atom<dim>> &atoms)
            : Function<dim>(1), atoms(atoms) {}

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void)component;
            double total_val = 0.0;
            for (const auto &atom : atoms)
            {
                const double r_c = atom.r_c;
                const double r_c_sq = r_c * r_c;
                const double r_c_8 = r_c_sq * r_c_sq * r_c_sq * r_c_sq;

                double r_sq = 0.0;
                for (unsigned int i = 0; i < dim; ++i)
                {
                    const double d = p[i] - atom.position[i];
                    r_sq += d * d;
                }
                const double r = std::sqrt(r_sq);

                if (r <= r_c)
                {
                    const double n = -21.0 * (r - r_c) * (r - r_c) * (r - r_c) *
                                     (6.0 * r_sq + 3.0 * r * r_c + r_c_sq);
                    const double d = 5.0 * numbers::PI * r_c_8;
                    total_val += atom.charge * n / d;
                }
            }
            return total_val;
        }

    private:
        const std::vector<Atom<dim>> atoms;
    };

    template <int dim>
    class CoulombPotential : public Function<dim>
    {
    public:
        explicit CoulombPotential(const std::vector<Atom<dim>> &atoms)
            : Function<dim>(1), atoms(atoms) {}

        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            (void)component;
            double total_val = 0.0;
            for (const auto &atom : atoms)
            {
                double r_sq = 0.0;
                for (unsigned int i = 0; i < dim; ++i)
                {
                    const double d = p[i] - atom.position[i];
                    r_sq += d * d;
                }
                const double r = std::sqrt(r_sq);
                total_val += atom.charge / (4.0 * numbers::PI * std::max(r, 1e-12));
            }
            return total_val;
        }

    private:
        const std::vector<Atom<dim>> atoms;
    };

    void solve_generalized_eigenproblem(LAPACKFullMatrix<double> &K,
                                         LAPACKFullMatrix<double> &M,
                                         Vector<double> &eigenvalues_out,
                                         LAPACKFullMatrix<double> &S_out)
    {
        const unsigned int n = K.m();
        std::vector<Vector<double>> eigenvectors;
        eigenvalues_out.reinit(n);

        K.compute_generalized_eigenvalues_symmetric(
            M,
            -std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            0.0,
            eigenvalues_out,
            eigenvectors,
            1);
        
        // S is matrix whose columns are the eigenvectors. Each eigenvector is of size n.
        S_out.reinit(n, n);
        for (unsigned int j = 0; j < eigenvectors.size(); ++j)
            for (unsigned int i = 0; i < n; ++i)
                S_out(i, j) = eigenvectors[j][i];
    }

    void apply_axis_0(const LAPACKFullMatrix<double> &A, const Vector<double> &in, Vector<double> &out, unsigned int N, bool transpose = false)
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

    void apply_axis_1(const LAPACKFullMatrix<double> &A, const Vector<double> &in, Vector<double> &out, unsigned int N, bool transpose = false)
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

    void apply_axis_2(const LAPACKFullMatrix<double> &A, const Vector<double> &in, Vector<double> &out, unsigned int N, bool transpose = false)
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

    void apply_laplacian_3d(const LAPACKFullMatrix<double> &K, const LAPACKFullMatrix<double> &M, const Vector<double> &in, Vector<double> &out, unsigned int N)
    {
        Vector<double> t1(N * N * N), t2(N * N * N), term(N * N * N);
        out.reinit(N * N * N);

        apply_axis_0(K, in, t1, N); apply_axis_1(M, t1, t2, N); apply_axis_2(M, t2, term, N); out += term;
        apply_axis_0(M, in, t1, N); apply_axis_1(K, t1, t2, N); apply_axis_2(M, t2, term, N); out += term;
        apply_axis_0(M, in, t1, N); apply_axis_1(M, t1, t2, N); apply_axis_2(K, t2, term, N); out += term;
    }

    void fdm_solve(const Vector<double> &f_mod, Vector<double> &u_0, unsigned int N, const LAPACKFullMatrix<double> &S, const Vector<double> &lambda)
    {
        Vector<double> temp1(N * N * N), temp2(N * N * N), F_tilde(N * N * N);

        apply_axis_0(S, f_mod, temp1, N, /*transpose=*/true);
        apply_axis_1(S, temp1, temp2, N, /*transpose=*/true);
        apply_axis_2(S, temp2, F_tilde, N, /*transpose=*/true);

        Vector<double> U_tilde(N * N * N);
        for (unsigned int k = 0; k < N; ++k)
            for (unsigned int j = 0; j < N; ++j)
                for (unsigned int i = 0; i < N; ++i)
                {
                    const unsigned int idx = i + j * N + k * N * N;
                    const double denom = lambda[i] + lambda[j] + lambda[k];
                    U_tilde[idx] = (std::abs(denom) > 1e-12) ? F_tilde[idx] / denom : 0.0;
                }

        u_0.reinit(N * N * N);
        apply_axis_2(S, U_tilde, temp1, N);
        apply_axis_1(S, temp1, temp2, N);
        apply_axis_0(S, temp2, u_0, N);
    }

    template <int dim, int fe_degree>
    class PoissonProblem
    {
    public:
        PoissonProblem(const unsigned int n_cells_per_edge_);
        void setup_system();
        void assemble_rhs();
        void assemble_system();
        void compute_interior_eigensystem();
        void solve();
        void run();

        double compute_electrostatic_energy() const;

    private:
        void build_lexicographic_map();

        MPI_Comm mpi_communicator;
        ConditionalOStream pcout;

        Triangulation<dim> triangulation;
        FE_Q<dim>          fe;
        DoFHandler<dim>    dof_handler;

        std::vector<Atom<dim>> atoms;
        unsigned int           n_cells_per_edge;
        double                 domain_start;
        double                 domain_end;

        LAPACKFullMatrix<double> K_1d, M_1d;
        LAPACKFullMatrix<double> S_1d_int;
        Vector<double>           eigenvalues_int;

        std::vector<types::global_dof_index> lex_to_dof;
        std::vector<Point<dim>>              support_points;

        Vector<double> system_rhs;
        Vector<double> solution;
    };

    template <int dim, int fe_degree>
    PoissonProblem<dim, fe_degree>::PoissonProblem(const unsigned int n_cells_per_edge_)
        : mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        , triangulation()
        , fe(fe_degree)
        , dof_handler(triangulation)
        , n_cells_per_edge(n_cells_per_edge_)
        , domain_start(-5.0)
        , domain_end(5.0)
    {
        Atom<dim> atom;
        atom.position = Point<dim>();
        atom.charge   = 12.0;
        atom.r_c      = 0.7;
        atoms.push_back(atom);
    }

    // TODO: Verify
    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::build_lexicographic_map()
    {
        static_assert(dim == 3, "Written for dim == 3.");
        const unsigned int n_dofs_1d = fe_degree * n_cells_per_edge + 1;

        support_points.assign(dof_handler.n_dofs(), Point<dim>());
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

        lex_to_dof.assign(n_dofs_1d * n_dofs_1d * n_dofs_1d, numbers::invalid_dof_index);

        const std::vector<unsigned int> lex_to_hier =
            FETools::lexicographic_to_hierarchic_numbering<3>(fe_degree);

        std::vector<types::global_dof_index> local_dofs(fe.dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            cell->get_dof_indices(local_dofs);

            const Point<dim> center = cell->center();
            const double h_cell = (domain_end - domain_start) / n_cells_per_edge;

            const unsigned int cx = static_cast<unsigned int>(std::round((center[0] - domain_start - 0.5 * h_cell) / h_cell));
            const unsigned int cy = static_cast<unsigned int>(std::round((center[1] - domain_start - 0.5 * h_cell) / h_cell));
            const unsigned int cz = static_cast<unsigned int>(std::round((center[2] - domain_start - 0.5 * h_cell) / h_cell));

            for (unsigned int lx = 0; lx <= fe_degree; ++lx)
                for (unsigned int ly = 0; ly <= fe_degree; ++ly)
                    for (unsigned int lz = 0; lz <= fe_degree; ++lz)
                    {
                        const unsigned int local_lex = lx + ly * (fe_degree + 1) + lz * (fe_degree + 1) * (fe_degree + 1);
                        const unsigned int local_hier = lex_to_hier[local_lex];

                        const unsigned int gx = cx * fe_degree + lx;
                        const unsigned int gy = cy * fe_degree + ly;
                        const unsigned int gz = cz * fe_degree + lz;

                        const unsigned int global_lex = gx + gy * n_dofs_1d + gz * n_dofs_1d * n_dofs_1d;
                        lex_to_dof[global_lex] = local_dofs[local_hier];
                    }
        }
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::setup_system()
    {
        GridGenerator::subdivided_hyper_cube(triangulation, n_cells_per_edge, domain_start, domain_end);
        dof_handler.distribute_dofs(fe);

        const unsigned int n_dofs_1d = fe_degree * n_cells_per_edge + 1;
        K_1d.reinit(n_dofs_1d, n_dofs_1d);
        M_1d.reinit(n_dofs_1d, n_dofs_1d);

        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());

        build_lexicographic_map();
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::assemble_rhs()
    {
        system_rhs = 0.0;
        const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
        FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const SmearedCharge<dim> rho(atoms);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            cell_rhs = 0.0;
            fe_values.reinit(cell);
            const auto &quadrature_points = fe_values.get_quadrature_points();

            for (unsigned int q_index = 0; q_index < fe_values.n_quadrature_points; ++q_index)
            {
                const double rho_q = rho.value(quadrature_points[q_index]);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += fe_values.shape_value(i, q_index) * rho_q * fe_values.JxW(q_index);
            }

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::assemble_system()
    {
        const unsigned int n_dofs_1d = fe_degree * n_cells_per_edge + 1;
        const QGauss<1> quadrature_formula(fe_degree + 1);
        const FE_Q<1> fe_1d(fe_degree);
        const unsigned int dofs_per_cell_1d = fe_1d.dofs_per_cell;

        const std::vector<unsigned int> lex_to_hierarchic =
            FETools::lexicographic_to_hierarchic_numbering<1>(fe_degree);

        FullMatrix<double> cell_K_1d(dofs_per_cell_1d, dofs_per_cell_1d);
        FullMatrix<double> cell_M_1d(dofs_per_cell_1d, dofs_per_cell_1d);

        const double h = (domain_end - domain_start) / static_cast<double>(n_cells_per_edge);

        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
            const Point<1> p_q = quadrature_formula.point(q);
            const double w_q   = quadrature_formula.weight(q);

            for (unsigned int i = 0; i < dofs_per_cell_1d; ++i)
            {
                const double phi_i  = fe_1d.shape_value(i, p_q);
                    const double dphi_i = fe_1d.shape_grad(i, p_q)[0];

                for (unsigned int j = 0; j < dofs_per_cell_1d; ++j)
                {
                    const double phi_j  = fe_1d.shape_value(j, p_q);
                    const double dphi_j = fe_1d.shape_grad(j, p_q)[0];

                    // Physical integration in 1D: map reference -> physical element
                    // d/dx = (1/h) d/dξ and dx = h dξ (unit reference [0,1])
                    cell_K_1d(i, j) += dphi_i * (1.0 / h) * dphi_j * (1.0 / h) * w_q * h;
                    cell_M_1d(i, j) += phi_i * phi_j * w_q * h;
                }
            }
        }

        K_1d = 0.0;
        M_1d = 0.0;

        for (unsigned int e = 0; e < n_cells_per_edge; ++e)
        {
            const unsigned int offset = e * fe_degree;
            for (unsigned int i_lex = 0; i_lex < dofs_per_cell_1d; ++i_lex)
            {
                const unsigned int i_hier = lex_to_hierarchic[i_lex];
                for (unsigned int j_lex = 0; j_lex < dofs_per_cell_1d; ++j_lex)
                {
                    const unsigned int j_hier = lex_to_hierarchic[j_lex];
                    K_1d(offset + i_lex, offset + j_lex) += cell_K_1d(i_hier, j_hier);
                    M_1d(offset + i_lex, offset + j_lex) += cell_M_1d(i_hier, j_hier);
                }
            }
        }

        compute_interior_eigensystem();
        assemble_rhs();
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::compute_interior_eigensystem()
    {
        const unsigned int n_dofs_1d = fe_degree * n_cells_per_edge + 1;
        const unsigned int n_int     = n_dofs_1d - 2;

        LAPACKFullMatrix<double> K_int(n_int, n_int);
        LAPACKFullMatrix<double> M_int(n_int, n_int);

        for (unsigned int i = 0; i < n_int; ++i)
            for (unsigned int j = 0; j < n_int; ++j)
            {
                K_int(i, j) = K_1d(i + 1, j + 1);
                M_int(i, j) = M_1d(i + 1, j + 1);
            }

        LAPACKFullMatrix<double> K_copy(K_int);
        LAPACKFullMatrix<double> M_copy(M_int);

        solve_generalized_eigenproblem(K_copy, M_copy, eigenvalues_int, S_1d_int);
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::solve()
    {
        const unsigned int n_dofs_1d = fe_degree * n_cells_per_edge + 1;
        const unsigned int n_int     = n_dofs_1d - 2;
        const unsigned int N3        = n_dofs_1d * n_dofs_1d * n_dofs_1d;

        const CoulombPotential<dim> coulomb(atoms);
        Vector<double> g_lex(N3);
        g_lex = 0.0;

        // Lifting the boundary condition: g_lex is the Coulomb potential on the boundary DOFs, zero elsewhere.

        for (unsigned int k = 0; k < n_dofs_1d; ++k)
            for (unsigned int j = 0; j < n_dofs_1d; ++j)
                for (unsigned int i = 0; i < n_dofs_1d; ++i)
                {
                    const bool on_boundary = (i == 0 || i == n_dofs_1d - 1 ||
                                              j == 0 || j == n_dofs_1d - 1 ||
                                              k == 0 || k == n_dofs_1d - 1);
                    if (!on_boundary)
                        continue;

                    const unsigned int lex = i + j * n_dofs_1d + k * n_dofs_1d * n_dofs_1d;
                    g_lex[lex] = coulomb.value(support_points[lex_to_dof[lex]]);
                }

        Vector<double> f_lex(N3);
        for (unsigned int lex = 0; lex < N3; ++lex)
            f_lex[lex] = system_rhs[lex_to_dof[lex]];

        Vector<double> Kg_lex;
        apply_laplacian_3d(K_1d, M_1d, g_lex, Kg_lex, n_dofs_1d);

        Vector<double> rhs_lex(N3);
        for (unsigned int lex = 0; lex < N3; ++lex)
            rhs_lex[lex] = f_lex[lex] - Kg_lex[lex];

        Vector<double> f_int(n_int * n_int * n_int);
        for (unsigned int k = 0; k < n_int; ++k)
            for (unsigned int j = 0; j < n_int; ++j)
                for (unsigned int i = 0; i < n_int; ++i)
                {
                    const unsigned int lex_full = (i + 1) + (j + 1) * n_dofs_1d + (k + 1) * n_dofs_1d * n_dofs_1d;
                    const unsigned int lex_int  = i + j * n_int + k * n_int * n_int;
                    f_int[lex_int] = rhs_lex[lex_full];
                }

        Vector<double> u_int;
        fdm_solve(f_int, u_int, n_int, S_1d_int, eigenvalues_int);

        Vector<double> u_lex(g_lex);
        for (unsigned int k = 0; k < n_int; ++k)
            for (unsigned int j = 0; j < n_int; ++j)
                for (unsigned int i = 0; i < n_int; ++i)
                {
                    const unsigned int lex_full = (i + 1) + (j + 1) * n_dofs_1d + (k + 1) * n_dofs_1d * n_dofs_1d;
                    const unsigned int lex_int  = i + j * n_int + k * n_int * n_int;
                    u_lex[lex_full] += u_int[lex_int];
                }

        solution.reinit(dof_handler.n_dofs());
        for (unsigned int lex = 0; lex < N3; ++lex)
            solution[lex_to_dof[lex]] = u_lex[lex];
    }

    template <int dim, int fe_degree>
    double PoissonProblem<dim, fe_degree>::compute_electrostatic_energy() const
    {
        const SmearedCharge<dim> forcing_function(atoms);
        double total_energy = 0.0;

        const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
        FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);

        std::vector<double> solution_values(quadrature_formula.size());

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            const auto &quadrature_points = fe_values.get_quadrature_points();
            fe_values.get_function_values(solution, solution_values);

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                const double phi_q = solution_values[q_index];
                const double rho_q = forcing_function.value(quadrature_points[q_index]);
                total_energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index);
            }
        }
        return total_energy;
    }

    template <int dim, int fe_degree>
    void PoissonProblem<dim, fe_degree>::run()
    {
        setup_system();
        assemble_system();
        solve();
        pcout << "Electrostatic energy: " << compute_electrostatic_energy() << std::endl;
    }
}

// int main(int argc, char *argv[])
// {
//     dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);


//     for (int i = 8; i <= 24; i += 4)
//     {
//         const auto start = std::chrono::high_resolution_clock::now();
//         FastDiagonalization::PoissonProblem<3, 3> poisson_problem(i);
//         std::cout << "Running with " << i << " cells per edge..." << std::endl;
//         poisson_problem.run();
//         const auto end = std::chrono::high_resolution_clock::now();
//         const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//         std::cout << "Time: " << duration.count() << " ms" << std::endl;
//     }
//     return 0;
// }

#define method_name "FastDiagonalization"
#define fe_degree 3

int main(int argc, char *argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

    try
    {
        const std::vector<int> cells_per_edge_list = {8, 12, 16, 20, 24};
        const std::string output_dir = "benchmark_results";
        const std::string csv_path   = output_dir + "/performance_study.csv";

        std::filesystem::create_directories(output_dir);
            
        // Check if file exists to determine if we write headers
        bool file_exists = std::filesystem::exists(csv_path);
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists)
        {
            csv << "method,fe_degree,cells_per_edge,total_cells,dofs,"
                << "setup_time_sec,assemble_time_sec,solve_time_sec,"
                << "total_time_sec,cg_iterations,time_per_iter_sec,energy\n";
        }

        for(int n_cells_per_edge : cells_per_edge_list)
        {
            std::cout << "\n=====================================\n"
                      << "Backend: " << method_name << " | FE Degree: " << fe_degree << "\n"
                      << "Running mesh with " << n_cells_per_edge << " cells per edge\n"
                      << "=====================================\n";

            FastDiagonalization::PoissonProblem<3, fe_degree> poisson_problem(n_cells_per_edge);

            // 1. Time Setup
            auto t0 = std::chrono::high_resolution_clock::now();
            poisson_problem.setup_system();
            auto t1 = std::chrono::high_resolution_clock::now();
            const double setup_time = std::chrono::duration<double>(t1 - t0).count();

            // 2. Time Assembly
            auto t2 = std::chrono::high_resolution_clock::now();
            poisson_problem.assemble_system();
            auto t3 = std::chrono::high_resolution_clock::now();
            const double assemble_time = std::chrono::duration<double>(t3 - t2).count();

            // 3. Time Solve
            auto t4 = std::chrono::high_resolution_clock::now();
            poisson_problem.solve();
            auto t5 = std::chrono::high_resolution_clock::now();
            const double solve_time = std::chrono::duration<double>(t5 - t4).count();

            const double total_time = setup_time + assemble_time + solve_time;

            const double energy = poisson_problem.compute_electrostatic_energy();

            // cells_per_edge, total_cells, dofs, setup_time, assemble_time, solve_time, total_time, cg_iterations, time_per_iter, energy

            csv << method_name << "," << fe_degree << "," << n_cells_per_edge << ","
                << n_cells_per_edge * n_cells_per_edge * n_cells_per_edge << ","
                // << poisson_problem.dof_handler.n_dofs() << ","
                << setup_time << "," << assemble_time << "," << solve_time << ","
                << total_time << ","
                << "N/A" << "," // CG iterations not applicable for this method
                << "N/A" << "," // Time per iteration not applicable for this method
                << energy
                << std::endl;
        }
        csv.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;

}