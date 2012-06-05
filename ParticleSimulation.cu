#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cstdlib>
#include <cstdio>


#include "WindData.hpp"
#include "Particles.hpp"
#include "ParticleSource.hpp"


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    WindData wd = WindDataFromASCII(argv[1]);
    //printWindData(wd);

    Particles p = Particles(num_particles);
    ParticlesRandomizePositions(p, 0, wd.num_x, 0, wd.num_y, 0, wd.num_z);
    //ParticlesPrintToVTK(p, std::cout);

    // copy wind data to gpu
    float4 *dev_wind;
    cudaMalloc((void **)&dev_wind, wd.data.size() * sizeof(float4));
    cudaMemcpy(dev_wind, &(wd.data[0]), wd.data.size() * sizeof(float4), cudaMemcpyHostToDevice);

    float4 *dev_particle_pos;
    cudaMalloc((void **)&dev_particle_pos, num_particles * sizeof(float4));
    cudaMemcpy(dev_particle_pos, &(p.position[0]), num_particles * sizeof(float4), cudaMemcpyHostToDevice);

    for (int i=0; i<100; i++) {
        advect_particles<<<128, 256>>>(
                0,
                dev_particle_pos, num_particles,
                dev_wind, wd.num_x, wd.num_y, wd.num_z, wd.num_t
        );

        char ofname[256];
        std::snprintf(ofname, 255, "output_%03d.particles", i);
        std::ofstream out(ofname);

        // copy particles back to host
        cudaMemcpy(&(p.position[0]), dev_particle_pos, num_particles * sizeof(float4), cudaMemcpyDeviceToHost);
        printParticlesVTK(p, out);
        out.close();
    }

    // copy particles back to host
    cudaMemcpy(&(p.position[0]), dev_particle_pos, num_particles * sizeof(float4), cudaMemcpyDeviceToHost);

    cudaFree(dev_wind);
    cudaFree(dev_particle_pos);

    return 0;
}

