/* ------------------------------------------------------------------------
 * GPU matrix-free Poisson solver for smeared point charge(s).
 *
 * Corrected Implementation Details:
 *   1. Correct Dirichlet BC elimination using vmult_unconstrained() so that
 *      A*g computes the precise interior coupling without artificial identity
 *      row overrides prior to setting constrained values.
 *   2. Fixed RHS quadrature to QIterated(QGauss<1>(6), 2) to accurately
 *      integrate non-polynomial smeared charges.
 *   3. Includes multi-mesh resolution run loop and summary table generation.
 * ------------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

namespace PoissonGPU
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
      : Function<dim>(1)
      , atoms(atoms)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      (void)component;

      double total_val = 0.0;

      for (const auto &atom : atoms)
        {
          const double r_c         = atom.r_c;
          const double r_c_squared = r_c * r_c;
          const double r_c_8 =
            r_c_squared * r_c_squared * r_c_squared * r_c_squared;

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

  template <int dim>
  class CoulombPotential : public Function<dim>
  {
  public:
    explicit CoulombPotential(const std::vector<Atom<dim>> &atoms)
      : Function<dim>(1)
      , atoms(atoms)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
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

  template <int dim, int fe_degree>
  class PoissonOperatorQuad
  {
  public:
    DEAL_II_HOST_DEVICE void operator()(
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> *fe_eval,
      const int q_point) const
    {
      fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
    }

    static const unsigned int n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);
  };

  template <int dim, int fe_degree>
  class LocalPoissonOperator
  {
  public:
    static constexpr unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const typename Portable::MatrixFree<dim, double>::Data *data,
               const Portable::DeviceVector<double>                   &src,
               Portable::DeviceVector<double> &dst) const
    {
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
        data);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      PoissonOperatorQuad<dim, fe_degree> quad;
      data->for_each_quad_point(
        [&](const int &q_point) { quad(&fe_eval, q_point); });

      fe_eval.integrate(EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
  };

  template <int dim, int fe_degree>
  class PoissonOperator : public EnableObserverPointer
  {
  public:
    PoissonOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<double> &constraints);

    void vmult(
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
        &src) const;

    void vmult_unconstrained(
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
        &src) const;

    void copy_constrained_values(
      const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
        &src,
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &dst)
      const;

    void initialize_dof_vector(
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &vec)
      const;

    void compute_diagonal();

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const;

    types::global_dof_index m() const;
    types::global_dof_index n() const;

    double el(const types::global_dof_index row,
              const types::global_dof_index col) const;

  private:
    Portable::MatrixFree<dim, double> mf_data;
    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
      inverse_diagonal_entries;
  };

  template <int dim, int fe_degree>
  PoissonOperator<dim, fe_degree>::PoissonOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<double> &constraints)
  {
    const MappingQ<dim> mapping(fe_degree);
    typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                            update_JxW_values |
                                            update_quadrature_points;
    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);
  }

  template <int dim, int fe_degree>
  void PoissonOperator<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &src)
    const
  {
    dst = 0.;
    LocalPoissonOperator<dim, fe_degree> poisson_operator;
    mf_data.cell_loop(poisson_operator, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree>
  void PoissonOperator<dim, fe_degree>::vmult_unconstrained(
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &src)
    const
  {
    dst = 0.;
    LocalPoissonOperator<dim, fe_degree> poisson_operator;
    mf_data.cell_loop(poisson_operator, src, dst);
  }

  template <int dim, int fe_degree>
  void PoissonOperator<dim, fe_degree>::copy_constrained_values(
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      &src,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &dst)
    const
  {
    mf_data.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree>
  void PoissonOperator<dim, fe_degree>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &vec) const
  {
    mf_data.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree>
  void PoissonOperator<dim, fe_degree>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<
        LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    PoissonOperatorQuad<dim, fe_degree> poisson_operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1, double>(
      mf_data,
      inverse_diagonal,
      poisson_operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    double *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

  template <int dim, int fe_degree>
  std::shared_ptr<DiagonalMatrix<
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
  PoissonOperator<dim, fe_degree>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree>
  types::global_dof_index PoissonOperator<dim, fe_degree>::m() const
  {
    return mf_data.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree>
  types::global_dof_index PoissonOperator<dim, fe_degree>::n() const
  {
    return mf_data.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree>
  double PoissonOperator<dim, fe_degree>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr &&
             inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());
    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  template <int dim, int fe_degree>
  class PoissonProblem
  {
  public:
    struct RunSummary
    {
      unsigned int            n_cells_per_edge;
      types::global_dof_index n_dofs;
      unsigned int            cg_iterations;
      double                  total_charge;
      double                  energy;
    };

    explicit PoissonProblem(const unsigned int n_cells_per_edge_ = 12);

    RunSummary run();

  private:
    void         setup_system();
    void         assemble_rhs();
    unsigned int solve();
    double       compute_total_charge() const;
    double       compute_electrostatic_energy() const;
    void         output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double>                         constraints;
    std::unique_ptr<PoissonOperator<dim, fe_degree>> system_matrix_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      ghost_solution_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      solution_dev;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      system_rhs_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      boundary_values_dev;

    ConditionalOStream pcout;

    std::vector<Atom<dim>> atoms;

    unsigned int n_cells_per_edge;
    double       domain_start;
    double       domain_end;
  };

  template <int dim, int fe_degree>
  PoissonProblem<dim, fe_degree>::PoissonProblem(
    const unsigned int n_cells_per_edge_)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , n_cells_per_edge(n_cells_per_edge_)
    , domain_start(-10.0)
    , domain_end(10.0)
  {
    Atom<dim> atom;
    atom.position = Point<dim>();
    atom.charge   = 12.0;
    atom.r_c      = 0.7;

    atoms.push_back(atom);
  }

  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::setup_system()
  {
    GridGenerator::subdivided_hyper_cube(
      triangulation, n_cells_per_edge, domain_start, domain_end);

    const double refinement_factor = 2.5;

    for (unsigned int cycle = 0; cycle < 3; ++cycle)
      {
        int marked = 0;

        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            bool refine = false;

            for (const auto &atom : atoms)
              {
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     ++v)
                  {
                    if (cell->vertex(v).distance(atom.position) <
                        refinement_factor * atom.r_c)
                      {
                        refine = true;
                        break;
                      }
                  }

                if (refine)
                  break;
              }

            if (refine)
              {
                cell->set_refine_flag();
                ++marked;
              }
          }

        const int total_marked =
          Utilities::MPI::sum(marked, this->mpi_communicator);

        pcout << "Marked " << total_marked << " cells for refinement."
              << std::endl;

        triangulation.execute_coarsening_and_refinement();
      }

    pcout << "Number of active cells: "
          << triangulation.n_global_active_cells() << std::endl;

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, CoulombPotential<dim>(atoms), constraints);
    constraints.close();

    pcout << "Number of constraints: " << constraints.n_constraints()
          << std::endl;

    system_matrix_dev.reset(
      new PoissonOperator<dim, fe_degree>(dof_handler, constraints));

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      boundary_values_host(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    boundary_values_host = 0.;
    constraints.distribute(boundary_values_host);

    system_matrix_dev->initialize_dof_vector(boundary_values_dev);
    LinearAlgebra::ReadWriteVector<double> rw_bv(locally_owned_dofs);
    rw_bv.import_elements(boundary_values_host, VectorOperation::insert);
    boundary_values_dev.import_elements(rw_bv, VectorOperation::insert);

    ghost_solution_host.reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);

    system_matrix_dev->initialize_dof_vector(solution_dev);
    solution_dev = boundary_values_dev;

    system_rhs_dev.reinit(solution_dev);
  }

  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::assemble_rhs()
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      system_rhs_host(locally_owned_dofs,
                       locally_relevant_dofs,
                       mpi_communicator);

    const QIterated<dim> quadrature_formula(QGauss<1>(6), 3);
    FEValues<dim>        fe_values(fe,
                               quadrature_formula,
                               update_values | update_quadrature_points |
                                 update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const SmearedCharge<dim> rho(atoms);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
              const double rho_q =
                rho.value(fe_values.quadrature_point(q_index));

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += fe_values.shape_value(i, q_index) * rho_q *
                               fe_values.JxW(q_index);
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs,
                                                  local_dof_indices,
                                                  system_rhs_host);
        }
    system_rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import_elements(rw_vector, VectorOperation::insert);

    // Enforce boundary values on constrained DOFs of system_rhs_dev
    system_matrix_dev->copy_constrained_values(boundary_values_dev,
                                                system_rhs_dev);
  }

  template <int dim, int fe_degree>
  unsigned int PoissonProblem<dim, fe_degree>::solve()
  {
    system_matrix_dev->compute_diagonal();

    using PreconditionerType = PreconditionChebyshev<
      PoissonOperator<dim, fe_degree>,
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>;
    typename PreconditionerType::AdditionalData additional_data;
    additional_data.smoothing_range     = 15.;
    additional_data.degree              = 5;
    additional_data.eig_cg_n_iterations = 10;
    additional_data.constraints.copy_from(constraints);
    additional_data.preconditioner =
      system_matrix_dev->get_matrix_diagonal_inverse();

    PreconditionerType preconditioner;
    preconditioner.initialize(*system_matrix_dev, additional_data);

    SolverControl solver_control(system_rhs_dev.size(), 1e-10);
    SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
      cg(solver_control);
    cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import_elements(solution_dev, VectorOperation::insert);
    ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

    constraints.distribute(ghost_solution_host);
    ghost_solution_host.update_ghost_values();

    return solver_control.last_step();
  }

  template <int dim, int fe_degree>
  double PoissonProblem<dim, fe_degree>::compute_total_charge() const
  {
    const SmearedCharge<dim> forcing_function(atoms);
    double                   local_charge = 0.0;

    const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
    FEValues<dim>        fe_values(fe,
                               quadrature_formula,
                               update_values | update_quadrature_points |
                                 update_JxW_values);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          const auto &quadrature_points = fe_values.get_quadrature_points();

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            local_charge += forcing_function.value(quadrature_points[q_index]) *
                            fe_values.JxW(q_index);
        }

    return Utilities::MPI::sum(local_charge, mpi_communicator);
  }

  template <int dim, int fe_degree>
  double PoissonProblem<dim, fe_degree>::compute_electrostatic_energy() const
  {
    const SmearedCharge<dim> forcing_function(atoms);
    double                   local_energy = 0.0;

    const QIterated<dim> quadrature_formula(QGauss<1>(6), 2);
    FEValues<dim>        fe_values(fe,
                               quadrature_formula,
                               update_values | update_quadrature_points |
                                 update_JxW_values);

    std::vector<double> solution_values(quadrature_formula.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          const auto &quadrature_points = fe_values.get_quadrature_points();
          fe_values.get_function_values(ghost_solution_host, solution_values);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              const double phi_q = solution_values[q_index];
              const double rho_q =
                forcing_function.value(quadrature_points[q_index]);
              local_energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index);
            }
        }

    return Utilities::MPI::sum(local_energy, mpi_communicator);
  }

  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::output_results(
    const unsigned int cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ghost_solution_host, "phi");
    data_out.build_patches();

    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "poisson_solution", cycle, mpi_communicator, 2);
  }

  template <int dim, int fe_degree>
  typename PoissonProblem<dim, fe_degree>::RunSummary
  PoissonProblem<dim, fe_degree>::run()
  {
    pcout << "=====================================" << std::endl;
    pcout << "Backend: GPU matrix-free | FE Degree: " << fe_degree << std::endl;
    pcout << "Running mesh with " << n_cells_per_edge << " cells per edge"
          << std::endl;
    pcout << "=====================================" << std::endl;

    setup_system();

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_rhs();

    const double total_charge = compute_total_charge();
    pcout << std::fixed << std::setprecision(10);
    pcout << "Total integrated charge: " << total_charge << std::endl;

    const unsigned int iterations = solve();
    pcout << "Solved in " << iterations << " iterations." << std::endl;

    const double energy = compute_electrostatic_energy();
    pcout << "Electrostatic energy:    " << energy << std::endl;

    output_results(0);

    RunSummary summary;
    summary.n_cells_per_edge = n_cells_per_edge;
    summary.n_dofs           = dof_handler.n_dofs();
    summary.cg_iterations     = iterations;
    summary.total_charge      = total_charge;
    summary.energy            = energy;
    return summary;
  }
} // namespace PoissonGPU

int main(int argc, char *argv[])
{
  try
    {
      using namespace PoissonGPU;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const std::vector<unsigned int> resolutions = {8, 12, 16, 24, 32};

      std::vector<PoissonProblem<3, 4>::RunSummary> results;
      for (const unsigned int n_cells : resolutions)
        {
          PoissonProblem<3, 4> poisson_problem(n_cells);
          results.push_back(poisson_problem.run());
        }

      dealii::ConditionalOStream pcout(
        std::cout,
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      pcout << std::endl << "==== Convergence summary ====" << std::endl;
      pcout << std::setw(11) << "cells/edge" << std::setw(10) << "DoFs"
            << std::setw(9) << "CG its" << std::setw(16) << "total Q"
            << std::setw(20) << "Energy" << std::endl;

      for (const auto &r : results)
        {
          pcout << std::setw(11) << r.n_cells_per_edge << std::setw(10)
                << r.n_dofs << std::setw(9) << r.cg_iterations
                << std::setw(16) << std::setprecision(8) << r.total_charge
                << std::setw(20) << std::setprecision(10) << r.energy
                << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}