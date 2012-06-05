#include <cstdio>


#include "ParticleSource.hpp"
#include "Particles.cuh"
#include "WindData.cuh"


struct ParticleInitFunctor {
    ParticleInitFunctor(const ParticleSource &src_) : src(src_) { }

    ParticleSource src;

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple tup) const
    {
        float &x = thrust::get<0>(tup);
        float &y = thrust::get<1>(tup);
        float &z = thrust::get<2>(tup);

        float x_d = x - (size_t)x;  // distance from point's x to prev x measurement
        float y_d = y - (size_t)y;
        float z_d = z - (size_t)z;

        size_t i000 = get_index(x+0, y+0, z+0, t);
        size_t i100 = get_index(x+1, y+1, z+0, t);
        size_t i010 = get_index(x+0, y+1, z+0, t);
        size_t i110 = get_index(x+1, y+1, z+0, t);
        size_t i001 = get_index(x+0, y+0, z+1, t);
        size_t i101 = get_index(x+1, y+0, z+1, t);
        size_t i011 = get_index(x+0, y+1, z+1, t);
        size_t i111 = get_index(x+1, y+1, z+1, t);

        float c00 = u[i000] * (1 - x_d) + u[i100] * x_d;
        float c10 = u[i010] * (1 - x_d) + u[i110] * x_d;
        float c01 = u[i001] * (1 - x_d) + u[i101] * x_d;
        float c11 = u[i011] * (1 - x_d) + u[i111] * x_d;

        float c0 = c00 * (1 - y_d) + c10 * y_d;
        float c1 = c01 * (1 - y_d) + c11 * y_d;

        // TODO: NOTE the 0.05f is equiv to conversion between velocity and cell size
        // TODO: will also have to interpolate between time steps as well
        x += 0.05f * (c0 * (1 - z_d) + c1 * z_d);

        c00 = v[i000] * (1 - x_d) + v[i100] * x_d;
        c10 = v[i010] * (1 - x_d) + v[i110] * x_d;
        c01 = v[i001] * (1 - x_d) + v[i101] * x_d;
        c11 = v[i011] * (1 - x_d) + v[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        y += 0.05f * (c0 * (1 - z_d) + c1 * z_d);

        c00 = w[i000] * (1 - x_d) + w[i100] * x_d;
        c10 = w[i010] * (1 - x_d) + w[i110] * x_d;
        c01 = w[i001] * (1 - x_d) + w[i101] * x_d;
        c11 = w[i011] * (1 - x_d) + w[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        z += 0.05f * (c0 * (1 - z_d) + c1 * z_d);

        if (x < 0 || x > num_x)
            x = num_x / 2;
        if (y < 0 || y > num_y)
            y = num_y / 2;
        if (z < 0 || z > num_z)
            z = num_z / 2;
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
