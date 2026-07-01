#include "atom_system.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

namespace Physics
{
    template <int dim>
    void AtomSystem<dim>::add_atom(const Atom<dim> &atom)
    {
        atoms.push_back(atom);
    }

    template <int dim>
    void AtomSystem<dim>::set_atoms(const std::vector<Atom<dim>> &atoms_)
    {
        atoms = atoms_;
    }

    template <int dim>
    const std::vector<Atom<dim>> &AtomSystem<dim>::get_atoms() const
    {
        return atoms;
    }

    template <int dim>
    SmearedCharge<dim> AtomSystem<dim>::get_charge_density() const
    {
        return SmearedCharge<dim>(atoms);
    }

    template <int dim>
    CoulombPotential<dim> AtomSystem<dim>::get_boundary_potential() const
    {
        return CoulombPotential<dim>(atoms);
    }

    template <int dim>
    double AtomSystem<dim>::total_charge(dealii::DoFHandler<dim> &dof_handler,
                                          dealii::FE_Q<dim>       &fe) const
    {
        using namespace dealii;

        SmearedCharge<dim> forcing_function(atoms);

        double local_charge = 0.0;

        QIterated<dim> quad(QGauss<1>(6), 2); // TODO: make quadrature order a parameter
        FEValues<dim> fe_values(fe, quad,
                                 update_values | update_JxW_values | update_quadrature_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            // Only the owning rank integrates a given cell to avoid duplicated contributions.
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                const std::vector<Point<dim>> &quadrature_points = fe_values.get_quadrature_points();

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    local_charge += forcing_function.value(quadrature_points[q_index]) *
                                     fe_values.JxW(q_index);
                }
            }
        }

        // Reduce the local charge contributions from every rank to one global scalar integral.
        return Utilities::MPI::sum(local_charge, MPI_COMM_WORLD);
    }

    template <int dim>
    double AtomSystem<dim>::electrostatic_energy(
        dealii::DoFHandler<dim>                            &dof_handler,
        dealii::FE_Q<dim>                                   &fe,
        const dealii::TrilinosWrappers::MPI::Vector         &solution) const
    {
        using namespace dealii;

        SmearedCharge<dim> forcing_function(atoms);

        double local_energy = 0.0;

        QIterated<dim> quad(QGauss<1>(6), 2);
        FEValues<dim> fe_values(fe, quad,
                                 update_values | update_JxW_values | update_quadrature_points);

        // Build a ghosted copy of the solved vector so each rank can read
        // neighbor-owned DoFs during interpolation.
        IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        TrilinosWrappers::MPI::Vector solution_with_ghosts;
        solution_with_ghosts.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_with_ghosts = solution;
        solution_with_ghosts.update_ghost_values();

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            // Only the owning rank evaluates the energy contribution for each cell.
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);

                const std::vector<Point<dim>> &quadrature_points = fe_values.get_quadrature_points();
                std::vector<double> solution_values(fe_values.n_quadrature_points);
                fe_values.get_function_values(solution_with_ghosts, solution_values);

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    const double phi_q = solution_values[q_index];
                    const double rho_q = forcing_function.value(quadrature_points[q_index]);
                    local_energy += 0.5 * rho_q * phi_q * fe_values.JxW(q_index);
                }
            }
        }

        // Reduce the local energy contributions from every rank to one global scalar integral.
        return Utilities::MPI::sum(local_energy, MPI_COMM_WORLD);
    }

    // ---------------- Explicit instantiations ----------------
    template class AtomSystem<2>;
    template class AtomSystem<3>;
}