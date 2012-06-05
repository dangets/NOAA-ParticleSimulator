
#include "Particles.cuh"


int main(int argc, const char *argv[])
{
    HostParticles   hp(100);
    DeviceParticles dp(100);

    ParticlesRandomizePosition(hp, 0, 64, 0, 64, 0, 16);
    hp.print(std::cout);

    // fill the host particles in sequence
    //ParticlesFillSequence(hp);
    // copy particles to device
    //dp = hp;
    //hp.print(std::cout);

    return 0;
}
