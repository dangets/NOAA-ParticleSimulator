#ifndef PARTICLESOURCE_HPP
#define PARTICLESOURCE_HPP

#include <string>
#include <ctime>
#include <stdexcept>

using std::time_t;
using std::size_t;


struct ParticleSource {
    std::string id;
    // particle_type...

    float pos_x;
    float pos_y;
    float pos_z;

    time_t release_start; // seconds
    time_t release_stop;  // seconds
    float  release_rate;  // particles / second

    float dx;
    float dy;
    float dz;

    // constructor
    ParticleSource(float x, float y, float z,
            time_t start, time_t stop, float rate,
            float dx=0.1f, float dy=0.1f, float dz=0.1f) :
        pos_x(x), pos_y(y), pos_z(z),
        release_start(start), release_stop(stop), release_rate(rate),
        dx(dx), dy(dy), dz(dz)
    {
        if (release_start >= release_stop) {
            throw std::invalid_argument("stop must be > start");
        }
        if (release_rate < 0) {
            throw std::invalid_argument("release rate must be positive");
        }
    }

    // copy constructor
    ParticleSource(const ParticleSource &ps) :
        pos_x(ps.pos_x), pos_y(ps.pos_y), pos_z(ps.pos_z),
        release_start(ps.release_start), release_stop(ps.release_stop), release_rate(ps.release_rate),
        dx(ps.dx), dy(ps.dy), dz(ps.dz) { }

    // returns the number of particles released over source lifetime
    size_t lifetimeParticlesReleased() const {
        return (size_t) ((release_stop - release_start) * release_rate);
    }
};



#endif /* end of include guard: PARTICLESOURCE_HPP */
