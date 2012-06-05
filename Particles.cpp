
#include <iostream>
#include <iomanip>

#include "Particles.hpp"


void ParticlesPrintToVTK(const Particles &p, std::ostream &out)
{
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "junk title" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET POLYDATA" << std::endl;
    out << "POINTS " << p.length << " float" << std::endl;

    out << std::fixed << std::setprecision(3);

    for (size_t i=0; i<p.length; i++) {
        out << p.pos_x[i] << " "
            << p.pos_y[i] << " "
            << p.pos_z[i] << std::endl;
    }
}


void ParticlesRandomizePositions(Particles &p, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
    for (size_t i=0; i<p.length; i++) {
        p.pos_x[i] = xmax * (float)drand48();
        p.pos_y[i] = ymax * (float)drand48();
        p.pos_z[i] = zmax * (float)drand48();
    }
}


