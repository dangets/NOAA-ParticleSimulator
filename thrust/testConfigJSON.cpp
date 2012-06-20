#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "Config.hpp"
#include "ConfigJSON.hpp"
#include "ParticleSource.hpp"


void printConfig(const Config &cfg) {
    std::cout << "num_seconds: " << cfg.num_seconds << std::endl;
    for (size_t i=0; i<cfg.particle_sources.size(); ++i) {
        const ParticleSource &src = cfg.particle_sources[i];
        std::cout << src.id << "-----------" << std::endl;
        std::cout << "pos_x: " << src.pos_x << std::endl;
        std::cout << "pos_y: " << src.pos_y << std::endl;
        std::cout << "pos_z: " << src.pos_z << std::endl;

        std::cout << "dx: " << src.dx << std::endl;
        std::cout << "dy: " << src.dy << std::endl;
        std::cout << "dz: " << src.dz << std::endl;

        std::cout << "release_start: " << src.release_start << std::endl;
        std::cout << "release_stop: " << src.release_stop << std::endl;
        std::cout << "release_rate: " << src.release_rate << std::endl;
    }
}


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s <json-config>\n", argv[0]);
        std::exit(1);
    }

    std::ifstream cfg_in(argv[1]);
    Config cfg = ConfigFromJSON(cfg_in);

    printConfig(cfg);

    return 0;
}
