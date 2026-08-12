#ifndef PHYSICS_NUCLEAR_CHARGE_DENSITY_H
#define PHYSICS_NUCLEAR_CHARGE_DENSITY_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

#include "atom.h"

namespace Physics
{
    template <int dim>
    class SmearedCharge : public dealii::Function<dim>
    {
    public:
        SmearedCharge() = default;
        explicit SmearedCharge(const std::vector<Atom<dim>> &atoms);

        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override;

        void set_atoms(const std::vector<Atom<dim>> &atoms);

    private:
        double single_atom_value(const dealii::Point<dim> &p,
                                  const Atom<dim> &atom) const;

        std::vector<Atom<dim>> atoms;
    };

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