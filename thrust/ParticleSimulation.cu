#include <cstdio>


#include "ParticleSource.hpp"
#include "Particles.cuh"
#include "WindData.cuh"


struct ParticleInitFunctor {
    ParticleInitFunctor(const ParticleSource &src_) : src(src_) { }

    ParticleSource src;

    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // where t = (p.x, p.y, p.z, p.birthtime)

        thrust::get<3>(t)

        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};

int main(int argc, const char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    HostWindData wd = WindDataFromASCII(argv[1]);

    // create particle source
    ParticleSource src = ParticleSource(10, 10, 10, 0, 120, 10);

    // create particle array on GPU
    HostParticles   hp(src.lifetimeParticlesReleased());
    DeviceParticles dp(src.lifetimeParticlesReleased());

    // main loop

    return 0;
}
