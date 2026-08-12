#ifndef PHYSICS_ATOM_SYSTEM_H
#define PHYSICS_ATOM_SYSTEM_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

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

        SmearedCharge<dim>    get_charge_density() const;
        CoulombPotential<dim> get_boundary_potential() const;

        double total_charge(const dealii::DoFHandler<dim> &dof_handler,
                             const dealii::FE_Q<dim>       &fe,
                             const unsigned int             quad_degree=6) const;

        template <typename VectorType>
        double electrostatic_energy(
            const dealii::DoFHandler<dim> &dof_handler,
            const dealii::FE_Q<dim>       &fe,
            const VectorType              &solution,
            const unsigned int             quad_degree=6) const;

    private:
        std::vector<Atom<dim>> atoms;
    };
}

#endif