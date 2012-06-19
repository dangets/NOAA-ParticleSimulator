
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "ParticleSource.hpp"
#include "Particles.cuh"


int main(int argc, const char *argv[])
{
    ParticleSource src(11, 12, 13, 0, 8, 3);

    HostParticles hp(src.lifetimeParticlesReleased());
    DeviceParticles dp(src.lifetimeParticlesReleased());

    //ParticlesRandomizePosition(hp, 0, 64, 0, 64, 0, 16);
    //ParticlesPrint(hp, std::cout);
    //std::cout << "-----------------" << std::endl;

    ParticlesFillPosition(hp, src.pos_x, src.pos_y, src.pos_z);
    ParticlesFillBirthTime(hp, src.release_start, src.release_stop, src.release_rate);
    ParticlesPrint(hp, std::cout);
    std::cout << "-----------------" << std::endl;

    dp = hp;
    ParticlesPrint(dp, std::cout);
    std::cout << "-----------------" << std::endl;

    ParticlesPrintActive(hp, std::cout, 3);
    std::cout << "-----------------" << std::endl;

    return 0;
}
