#include "nuclear_charge_density.h"
#include <cmath>

namespace Physics
{
    namespace
    {
        constexpr double PI = 3.14159265358979323846;
    }

    // ---------------- SmearedCharge ----------------

    template <int dim>
    SmearedCharge<dim>::SmearedCharge(const std::vector<Atom<dim>> &atoms_)
        : atoms(atoms_)
    {}

    template <int dim>
    void SmearedCharge<dim>::set_atoms(const std::vector<Atom<dim>> &atoms_)
    {
        atoms = atoms_;
    }

    template <int dim>
    double SmearedCharge<dim>::single_atom_value(const dealii::Point<dim> &p,
                                                  const Atom<dim> &atom) const
    {
        const double r_c          = atom.r_c;
        const double r_c_squared  = r_c * r_c;
        const double r_c_8        = r_c_squared * r_c_squared * r_c_squared * r_c_squared;

        double r_squared = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double d = p[i] - atom.position[i];
            r_squared += d * d;
        }

        const double r = std::sqrt(r_squared);

        if (r <= r_c)
        {
            const double n = -21.0 * (r - r_c) * (r - r_c) * (r - r_c) *
                              (6.0 * r_squared + 3.0 * r * r_c + r_c_squared);
            const double d = 5.0 * PI * r_c_8;
            return atom.charge * n / d;
        }
        else
        {
            return 0.0;
        }
    }

    template <int dim>
    double SmearedCharge<dim>::value(const dealii::Point<dim> &p,
                                      const unsigned int component) const
    {
        (void) component;

        double total = 0.0;
        for (const auto &atom : atoms)
            total += single_atom_value(p, atom);

        return total;
    }

    // ---------------- CoulombPotential ----------------

    template <int dim>
    CoulombPotential<dim>::CoulombPotential(const std::vector<Atom<dim>> &atoms_)
        : atoms(atoms_)
    {}

    template <int dim>
    void CoulombPotential<dim>::set_atoms(const std::vector<Atom<dim>> &atoms_)
    {
        atoms = atoms_;
    }

    template <int dim>
    double CoulombPotential<dim>::single_atom_value(const dealii::Point<dim> &p,
                                                      const Atom<dim> &atom) const
    {
        double r_squared = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double d = p[i] - atom.position[i];
            r_squared += d * d;
        }

        const double r = std::sqrt(r_squared);

        if (r < 1e-12)
            return atom.charge / 1e-12; // TODO: same guard as original

        return atom.charge / (4.0 * M_PI * r);
    }

    template <int dim>
    double CoulombPotential<dim>::value(const dealii::Point<dim> &p,
                                         const unsigned int component) const
    {
        (void) component;

        double total = 0.0;
        for (const auto &atom : atoms)
            total += single_atom_value(p, atom);

        return total;
    }

    // ---------------- Explicit instantiations ----------------
    template class SmearedCharge<2>;
    template class SmearedCharge<3>;
    template class CoulombPotential<2>;
    template class CoulombPotential<3>;
}