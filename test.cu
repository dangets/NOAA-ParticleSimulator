#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <vector>


struct ParticleSource {
    int id;

    float pos_x;
    float pos_y;
    float pos_z;

    // particle_type...
};


struct Particles {
    Particles(size_t length_) :
        length(length_),
        position(length), velocity(length) { }

    size_t length;
    std::vector<float4> position;
    std::vector<float4> velocity;

    // type
    // source_id
    // age / birthtime
    // has_deposited
    // ...
};


void printParticlesVTK(const Particles &p, std::ostream &out)
{
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "junk title" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET POLYDATA" << std::endl;
    out << "POINTS " << p.length << " float" << std::endl;

    out << std::fixed << std::setprecision(3);

    for (size_t i=0; i<p.length; i++) {
        out //<< i << " "
            << p.position[i].x << " "
            << p.position[i].y << " "
            << p.position[i].z << std::endl;
    }
}


void randomizeParticlesPositions(Particles &p, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
    for (size_t i=0; i<p.length; i++) {
        p.position[i].x = xmax * (float)drand48();
        p.position[i].y = ymax * (float)drand48();
        p.position[i].z = zmax * (float)drand48();
    }
}


struct WindData {
    WindData(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        data(num_cells)
    { }

    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;
    size_t num_cells;

    std::vector<float4> data;
};


WindData WindDataFromASCII(const char * fname)
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

    WindData wd(num_x, num_y, num_z, num_t);

    for (size_t t=0; t<num_t; t++) {
        size_t t_offset = t * num_z * num_y * num_x;
        for (size_t z=0; z<num_z; z++) {
            size_t z_offset = z * num_y * num_x;
            for (size_t y=0; y<num_y; y++) {
                size_t y_offset = y * num_x;
                for (size_t x=0; x<num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    ins >> wd.data[offset].x;
                    ins >> wd.data[offset].y;
                    ins >> wd.data[offset].z;
                }
            }
        }
    }

    ins.close();

    return wd;
}


void printWindData(const WindData &wd)
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

                    std::printf("%7.2f ", wd.data[offset].x);
                    std::printf("%7.2f ", wd.data[offset].y);
                    std::printf("%7.2f ", wd.data[offset].z);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
}


__device__
size_t get_index(size_t x, size_t y, size_t z, size_t t,
        size_t num_x, size_t num_y, size_t num_z, size_t num_t)
{
    x = min((unsigned int)x, (unsigned int)num_x-1);
    y = min((unsigned int)y, (unsigned int)num_y-1);
    z = min((unsigned int)z, (unsigned int)num_z-1);
    t = min((unsigned int)t, (unsigned int)num_t-1);

    return x + y * num_x + z * num_y * num_x + t * num_x * num_y * num_z;
}



__global__
void advect_particles(float t, float4 *pos, size_t num_p,
        float4 *wind, size_t num_x, size_t num_y, size_t num_z, size_t num_t)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_p)
        return;

    float4 &p = pos[i];

    float x_d = p.x - (size_t)p.x;  // distance from point's x to prev x measurement
    float y_d = p.y - (size_t)p.y;
    float z_d = p.z - (size_t)p.z;

    size_t i000 = get_index(p.x+0, p.y+0, p.z+0, t, num_x, num_y, num_z, num_t);
    size_t i100 = get_index(p.x+1, p.y+1, p.z+0, t, num_x, num_y, num_z, num_t);
    size_t i010 = get_index(p.x+0, p.y+1, p.z+0, t, num_x, num_y, num_z, num_t);
    size_t i110 = get_index(p.x+1, p.y+1, p.z+0, t, num_x, num_y, num_z, num_t);
    size_t i001 = get_index(p.x+0, p.y+0, p.z+1, t, num_x, num_y, num_z, num_t);
    size_t i101 = get_index(p.x+1, p.y+0, p.z+1, t, num_x, num_y, num_z, num_t);
    size_t i011 = get_index(p.x+0, p.y+1, p.z+1, t, num_x, num_y, num_z, num_t);
    size_t i111 = get_index(p.x+1, p.y+1, p.z+1, t, num_x, num_y, num_z, num_t);

    float c00 = wind[i000].x * (1 - x_d) + wind[i100].x * x_d;
    float c10 = wind[i010].x * (1 - x_d) + wind[i110].x * x_d;
    float c01 = wind[i001].x * (1 - x_d) + wind[i101].x * x_d;
    float c11 = wind[i011].x * (1 - x_d) + wind[i111].x * x_d;

    float c0 = c00 * (1 - y_d) + c10 * y_d;
    float c1 = c01 * (1 - y_d) + c11 * y_d;

    float u = c0 * (1 - z_d) + c1 * z_d;

    c00 = wind[i000].y * (1 - x_d) + wind[i100].y * x_d;
    c10 = wind[i010].y * (1 - x_d) + wind[i110].y * x_d;
    c01 = wind[i001].y * (1 - x_d) + wind[i101].y * x_d;
    c11 = wind[i011].y * (1 - x_d) + wind[i111].y * x_d;

    c0 = c00 * (1 - y_d) + c10 * y_d;
    c1 = c01 * (1 - y_d) + c11 * y_d;

    float v = c0 * (1 - z_d) + c1 * z_d;

    c00 = wind[i000].z * (1 - x_d) + wind[i100].z * x_d;
    c10 = wind[i010].z * (1 - x_d) + wind[i110].z * x_d;
    c01 = wind[i001].z * (1 - x_d) + wind[i101].z * x_d;
    c11 = wind[i011].z * (1 - x_d) + wind[i111].z * x_d;

    c0 = c00 * (1 - y_d) + c10 * y_d;
    c1 = c01 * (1 - y_d) + c11 * y_d;

    float w = c0 * (1 - z_d) + c1 * z_d;

    u *= 0.05f;     // conversion from speed to cell size
    v *= 0.05f;
    w *= 0.05f;

    p.x += u;
    p.y += v;
    p.z += w;


    if (p.x < 0)     p.x = 0;
    if (p.x > num_x) p.x = num_x - 1;

    if (p.y < 0)     p.y = 0;
    if (p.y > num_y) p.y = num_y - 1;

    if (p.z < 0)     p.z = 0;
    if (p.z > num_z) p.z = num_z - 1;
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    size_t num_particles = 100;

    WindData wd = WindDataFromASCII(argv[1]);
    //printWindData(wd);

    Particles p = Particles(num_particles);
    randomizeParticlesPositions(p, 0, wd.num_x, 0, wd.num_y, 0, wd.num_z);
    //printParticlesVTK(p, std::cout);

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

