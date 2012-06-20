#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <vector>

#include "ParticleSource.hpp"

struct Config {
    std::vector<ParticleSource> particle_sources;
    size_t num_seconds;
};


#endif /* end of include guard: CONFIG_HPP */

