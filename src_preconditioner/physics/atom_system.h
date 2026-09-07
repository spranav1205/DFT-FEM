#ifndef PHYSICS_ATOM_SYSTEM_H
#define PHYSICS_ATOM_SYSTEM_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_vector.h>

#include <vector>

#include "atom.h"
#include "nuclear_charge_density.h"

namespace Physics
{
    // Owns the set of atoms/nuclei in the problem and exposes the derived
    // charge-density / boundary-potential functions plus physical
    // observables (total charge, electrostatic energy) computed from a
    // finite-element solution.
    template <int dim>
    class AtomSystem
    {
    public:
        AtomSystem() = default;

        void add_atom(const Atom<dim> &atom);
        void set_atoms(const std::vector<Atom<dim>> &atoms);
        const std::vector<Atom<dim>> &get_atoms() const;

        // Function objects derived from the current atom configuration.
        SmearedCharge<dim>    get_charge_density() const;
        CoulombPotential<dim> get_boundary_potential() const;

        // Total charge integrated over the mesh: sum_cells int rho dV
        double total_charge(dealii::DoFHandler<dim>       &dof_handler,
                             dealii::FE_Q<dim>             &fe) const;

        // Electrostatic energy: 0.5 * int rho(x) * phi(x) dV
        double electrostatic_energy(
            dealii::DoFHandler<dim>                     &dof_handler,
            dealii::FE_Q<dim>                            &fe,
            const dealii::TrilinosWrappers::MPI::Vector &solution) const;

    private:
        std::vector<Atom<dim>> atoms;
    };
}

#endif