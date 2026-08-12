#include "atom_system.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/la_parallel_vector.h>

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


    // This is a sanity check to ensure that the total charge integrates to the sum of the individual atom charges.
    template <int dim>
    double AtomSystem<dim>::total_charge(const dealii::DoFHandler<dim> &dof_handler,
                                          const dealii::FE_Q<dim>       &fe,
                                          const unsigned int             quad_degree) const
    {
        using namespace dealii;

        SmearedCharge<dim> forcing_function(atoms);

        double local_charge = 0.0;

        QGauss<dim>  quad(quad_degree);
        FEValues<dim> fe_values(fe, quad,
                                update_values | update_JxW_values | update_quadrature_points);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
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

        return Utilities::MPI::sum(local_charge, dof_handler.get_mpi_communicator());
    }

    template <int dim>
    template <typename VectorType>
    double AtomSystem<dim>::electrostatic_energy(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::FE_Q<dim>       &fe,
        const VectorType              &solution,
        const unsigned int             quad_degree) const
    {
        using namespace dealii;

        SmearedCharge<dim> forcing_function(atoms);

        double local_energy = 0.0;

        QGauss<dim>  quad(quad_degree);
        FEValues<dim> fe_values(fe, quad,
                                update_values | update_JxW_values | update_quadrature_points);

        // Build a ghosted copy of the solution vector so each rank can evaluate
        // neighbor-owned DoFs during cell interpolation.
        IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        VectorType solution_with_ghosts;
        solution_with_ghosts.reinit(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    dof_handler.get_mpi_communicator());
        solution_with_ghosts = solution;
        solution_with_ghosts.update_ghost_values();

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
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

        return Utilities::MPI::sum(local_energy, dof_handler.get_mpi_communicator());
    }

    // Explicit instantiations for 2D and 3D

    template class AtomSystem<2>;
    template class AtomSystem<3>;

    template double AtomSystem<2>::electrostatic_energy(
        const dealii::DoFHandler<2> &,
        const dealii::FE_Q<2> &,
        const dealii::LinearAlgebra::distributed::Vector<double> &,
        const unsigned int) const;

    template double AtomSystem<3>::electrostatic_energy(
        const dealii::DoFHandler<3> &,
        const dealii::FE_Q<3> &,
        const dealii::LinearAlgebra::distributed::Vector<double> &,
        const unsigned int) const;
}