#ifndef PHYSICS_NUCLEAR_CHARGE_DENSITY_H
#define PHYSICS_NUCLEAR_CHARGE_DENSITY_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

#include "atom.h"

namespace Physics
{
    // Smeared nuclear charge density (sum over all atoms), same closed-form
    // compactly-supported profile as in the original single-atom version.
    template <int dim>
    class SmearedCharge : public dealii::Function<dim>
    {
    public:
        SmearedCharge() = default;
        explicit SmearedCharge(const std::vector<Atom<dim>> &atoms);

        virtual double value(const dealii::Point<dim> &p,
                              const unsigned int component = 0) const override;

        void set_atoms(const std::vector<Atom<dim>> &atoms);

    private:
        double single_atom_value(const dealii::Point<dim> &p,
                                  const Atom<dim> &atom) const;

        std::vector<Atom<dim>> atoms;
    };

    // Point-charge Coulomb potential (sum over all atoms) — used as the
    // Dirichlet boundary condition on the outer domain boundary.
    template <int dim>
    class CoulombPotential : public dealii::Function<dim>
    {
    public:
        CoulombPotential() = default;
        explicit CoulombPotential(const std::vector<Atom<dim>> &atoms);

        virtual double value(const dealii::Point<dim> &p,
                              const unsigned int component = 0) const override;

        void set_atoms(const std::vector<Atom<dim>> &atoms);

    private:
        double single_atom_value(const dealii::Point<dim> &p,
                                  const Atom<dim> &atom) const;

        std::vector<Atom<dim>> atoms;
    };
}

#endif