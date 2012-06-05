#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include <cstdlib>
#include <vector>


struct Particles {
    Particles(std::size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length),
        vel_u(length), vel_v(length), vel_w(length) { }

    std::size_t length;

    std::vector<float> pos_x;
    std::vector<float> pos_y;
    std::vector<float> pos_z;

    std::vector<float> vel_u;
    std::vector<float> vel_v;
    std::vector<float> vel_w;

    // type
    // source_id
    // age / birthtime
    // has_deposited
    // ...
};


void ParticlesPrintToVTK(const Particles &p, std::ostream &out);
void ParticlesRandomizePositions(Particles &p,
        float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

#endif /* end of include guard: PARTICLES_HPP */
