#include <iostream>
#include <fstream>
#include <cstdio>

#include "Particles.cuh"
#include "WindData.cuh"

HostWindData WindDataFromASCII(const char * fname)
{
    std::ifstream ins;

    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;

    ins.open(fname);
    ins >> num_x;
    ins >> num_y;
    ins >> num_z;
    ins >> num_t;

    HostWindData wd(num_x, num_y, num_z, num_t);

    for (size_t t=0; t<num_t; t++) {
        size_t t_offset = t * num_z * num_y * num_x;
        for (size_t z=0; z<num_z; z++) {
            size_t z_offset = z * num_y * num_x;
            for (size_t y=0; y<num_y; y++) {
                size_t y_offset = y * num_x;
                for (size_t x=0; x<num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    ins >> wd.u[offset];
                    ins >> wd.v[offset];
                    ins >> wd.w[offset];
                }
            }
        }
    }

    ins.close();

    return wd;
}


template<typename WindData>
void WindDataPrint(const WindData &wd)
{
    std::printf("%lu %lu %lu %lu\n", wd.num_x, wd.num_y, wd.num_z, wd.num_t);

    for (size_t t=0; t<wd.num_t; t++) {
        size_t t_offset = t * wd.num_z * wd.num_y * wd.num_x;
        for (size_t z=0; z<wd.num_z; z++) {
            size_t z_offset = z * wd.num_y * wd.num_x;
            for (size_t y=0; y<wd.num_y; y++) {
                size_t y_offset = y * wd.num_x;
                for (size_t x=0; x<wd.num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    std::printf("%7.2f ", wd.u[offset]);
                    std::printf("%7.2f ", wd.v[offset]);
                    std::printf("%7.2f ", wd.w[offset]);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
}



int main(int argc, const char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    size_t num_particles = 30000;
    size_t num_timesteps = 100000;

    HostWindData wd_h = WindDataFromASCII(argv[1]);
    DeviceWindData wd_d(wd_h);

    HostParticles p_h(num_particles);
    DeviceParticles p_d(num_particles);

    std::cerr << "randomizing particle position" << std::endl;
    // randomize particle position on host
    ParticlesRandomizePosition(p_h, 0, wd_h.num_x, 0, wd_h.num_y, 0, wd_h.num_z);

    std::cerr << "copying particles from host to device" << std::endl;
    // copy host particles to device
    p_d = p_h;

    // run the simulation on host side
    //for (int i=0; i<num_timesteps; i++) {
    //    char ofname[256];
    //    std::snprintf(ofname, 255, "host_output_%04d.particles", i);
    //    std::ofstream out(ofname);

    //    advectParticles(p_h, wd_h, 0);
    //    ParticlesPrint(p_h, out);

    //    out.close();
    //}


    // run the simulation on device side
    for (int i=0; i<num_timesteps; i++) {
        //char ofname[256];
        //std::snprintf(ofname, 255, "device_output_%04d.particles", i);
        //std::ofstream out(ofname);

        advectParticles(p_d, wd_d, 0);
        //ParticlesPrint(p_d, out);
        //out.close();
    }

    return 0;
}
