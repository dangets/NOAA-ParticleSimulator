#ifndef PARTICLESOURCE_HPP
#define PARTICLESOURCE_HPP

#include <ctime>
#include <stdexcept>

using std::size_t
using std::time_t;


struct ParticleSource {
    int id;
    // particle_type...

    float pos_x;
    float pos_y;
    float pos_z;

    const time_t release_start; // seconds
    const time_t release_stop;  // seconds
    const float release_rate;   // particles / second

    // constructor
    ParticleSource(float x, float y, float z,
            time_t start, time_t stop, float rate) :
        pos_x(x), pos_y(y), pos_z(z),
        release_start(start), release_stop(stop), release_rate(rate)
    {
        if (release_start >= release_stop) {
            throw std::invalid_argument("stop must be > start");
        }
        if (release_rate < 0) {
            throw std::invalid_argument("release rate must be positive");
        }
    }

    /// returns the number of particles released over source lifetime
    size_t lifetimeParticlesReleased() {
        return (size_t) ((release_stop - release_start) * release_rate);
    }
};


#endif /* end of include guard: PARTICLESOURCE_HPP */
