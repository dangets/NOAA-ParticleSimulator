
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Particles.cuh"



int main(int argc, const char *argv[])
{
    Particles< thrust::host_vector<float> >   hp(16);
    Particles< thrust::device_vector<float> > dp(16);

    ParticlesRandomizePosition(hp, 0, 64, 0, 64, 0, 16);
    ParticlesPrint(hp, std::cout);
    std::cout << "-----------------" << std::endl;

    dp = hp;
    ParticlesPrint(dp, std::cout);
    std::cout << "-----------------" << std::endl;

    ParticlesFillPosition(hp, 1.1, 2.2, 3.3);
    ParticlesPrint(hp, std::cout);
    std::cout << "-----------------" << std::endl;

    // fill the host particles in sequence
    //ParticlesFillSequence(hp);
    // copy particles to device
    //dp = hp;
    //hp.print(std::cout);

    return 0;
}
