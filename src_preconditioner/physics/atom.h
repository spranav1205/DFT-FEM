#ifndef PHYSICS_ATOM_H
#define PHYSICS_ATOM_H

#include <deal.II/base/point.h>

namespace Physics
{
    // A single nuclear center: its position, its charge, and the smearing
    // radius r_c used by the smeared nuclear charge density.
    template <int dim>
    struct Atom
    {
        dealii::Point<dim> position;
        double charge = 1.0;
        double r_c    = 0.7; // must be < half the inter-nuclear distance

        Atom() = default;

        Atom(const dealii::Point<dim> &position_,
             double charge_,
             double r_c_ = 0.7)
            : position(position_), charge(charge_), r_c(r_c_)
        {}
    };
}

#endif