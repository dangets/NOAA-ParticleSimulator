#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <vector>


struct Particle {
    float pos_x;
    float pos_y;
    float pos_z;

    // bool has_deposited;
    // source_id
    // age
    // pollutant_type, mass, etc...

    Particle() { }
    Particle(float x, float y, float z) :
        pos_x(x), pos_y(y), pos_z(z) { }

    std::string toString() {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << "x:" << pos_x << "  ";
        os << "y:" << pos_y << "  ";
        os << "z:" << pos_z;
        return os.str();
    }
};


//struct Particles {
//    float4 *position;
//    float4 *velocity;
//
//    // type
//    // source_id
//    // age
//    // ...
//    size_t length;
//
//    Particles(int length_) : length(length_) { }
//    ~Particles() { }
//};

//struct DataBrick {
//    float x_start;
//    float x_step;
//    size_t x_length;
//
//    float y_start;
//    float y_step;
//    size_t y_length;
//
//    float z_start;
//    float z_step;
//    size_t z_length;
//
//    time_t t_start;
//    time_t t_step;
//    size_t t_length;
//
//    float *data;
//};

//float DataBrickGetValue(float x, float y, float z, time_t t)
//{
//    float diff = x - x_start;
//    float quot = diff / x_step;
//    size_t pos = (size_t)quot;
//    float rem = quot - pos;
//
//}



// particle positions
Particle *dev_particles;

// meteo grid info
size_t dev_grid_num_x;
size_t dev_grid_num_y;
size_t dev_grid_num_z;

// wind velocities
texture<float> texWindU;    // [t][z][y][x]
texture<float> texWindV;
texture<float> texWindW;

float *dev_windU;
float *dev_windV;
float *dev_windW;


struct WindData {
    WindData(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        windU(num_cells),
        windV(num_cells),
        windW(num_cells)
    { }

    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;
    size_t num_cells;

    std::vector<float> windU;   // [t][z][y][x] = float
    std::vector<float> windV;
    std::vector<float> windW;
};


WindData readASCIIDataFile(const char * fname)
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

                    ins >> wd.windU[offset];
                    ins >> wd.windV[offset];
                    ins >> wd.windW[offset];
                }
            }
        }
    }

    ins.close();

    return wd;
}


void printWindData(const WindData &wd)
{
    for (size_t t=0; t<wd.num_t; t++) {
        size_t t_offset = t * wd.num_z * wd.num_y * wd.num_x;
        for (size_t z=0; z<wd.num_z; z++) {
            size_t z_offset = z * wd.num_y * wd.num_x;
            for (size_t y=0; y<wd.num_y; y++) {
                size_t y_offset = y * wd.num_x;
                for (size_t x=0; x<wd.num_x; x++) {
                    size_t offset = x + y_offset + z_offset + t_offset;

                    std::printf("%7.2f ", wd.windU[offset]);
                    std::printf("%7.2f ", wd.windV[offset]);
                    std::printf("%7.2f ", wd.windW[offset]);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
}




//struct ParticleSource {
//    int id;
//
//    float pos_x;
//    float pos_y;
//    float pos_z;
//};




__global__
void advect_particles(Particle *particles, size_t num_particles,
        size_t grid_num_x, size_t grid_num_y, size_t grid_num_z)
{
    size_t i = threadIdx.x;
    if (i >= num_particles)
        return;

    size_t pos_offset =
        particles[i].pos_x +
        particles[i].pos_y * grid_num_x +
        particles[i].pos_z * grid_num_y * grid_num_z;

    float vel_u = tex1Dfetch(texWindU, pos_offset);
    float vel_v = tex1Dfetch(texWindV, pos_offset);
    float vel_w = tex1Dfetch(texWindW, pos_offset);

    particles[i].pos_x += vel_u;
    particles[i].pos_y += vel_v;
    particles[i].pos_z += vel_w;

    if (particles[i].pos_x < 0)
        particles[i].pos_x = 0;
    if (particles[i].pos_x >= grid_num_x)
        particles[i].pos_x = grid_num_x - 1;

    if (particles[i].pos_y < 0)
        particles[i].pos_y = 0;
    if (particles[i].pos_y >= grid_num_y)
        particles[i].pos_y = grid_num_y - 1;

    if (particles[i].pos_z < 0)
        particles[i].pos_z = 0;
    if (particles[i].pos_z >= grid_num_z)
        particles[i].pos_z = grid_num_z - 1;
}


void print_particles(Particle *particles, size_t num)
{
    for (int i=0; i<num; i++) {
        std::cout << particles[i].toString() << std::endl;
    }
    std::cout << "-----------------" << std::endl;
}


void print_dev_particles(size_t num_particles)
{
    size_t num_bytes = num_particles * sizeof(Particle);

    Particle *tmp = (Particle *)malloc(num_bytes);
    cudaMemcpy(tmp, dev_particles, num_bytes, cudaMemcpyDeviceToHost);
    print_particles(tmp, num_particles);
    free(tmp);
}


void init_particles(size_t n, size_t x, size_t y, size_t z)
{
    size_t num_bytes = n * sizeof(Particle);

    // allocate particle array
    cudaMalloc((void **)&dev_particles, num_bytes);

    // allocate an array on the host to copy to the device
    Particle *tmp = (Particle *)malloc(num_bytes);
    for (int i=0; i<n; i++) {
        tmp[i].pos_x = x + i;
        tmp[i].pos_y = y + i;
        tmp[i].pos_z = z + i;
    }

    cudaMemcpy(dev_particles, tmp, num_bytes, cudaMemcpyHostToDevice);

    free(tmp);
}


void init_grid(size_t num_x, size_t num_y, size_t num_z, size_t num_t)
{
    size_t num_cells = num_x * num_y * num_z * num_t;
    size_t num_bytes = num_cells * sizeof(float);

    //std::cout << "num_bytes: " << num_bytes << std::endl;

    // allocate the device arrays
    cudaMalloc((void **)&dev_windU, num_bytes);
    cudaMalloc((void **)&dev_windV, num_bytes);
    cudaMalloc((void **)&dev_windW, num_bytes);

    // bind the texture references to the arrays
    cudaBindTexture(NULL, texWindU, dev_windU, num_bytes);
    cudaBindTexture(NULL, texWindV, dev_windV, num_bytes);
    cudaBindTexture(NULL, texWindW, dev_windW, num_bytes);

    // initialize the device arrays with random data
    float *tmp_data = (float *)malloc(num_bytes);
    for (int i=0; i<num_cells; i++) {
        tmp_data[i] = (float)drand48();
    }

    cudaMemcpy(dev_windU, tmp_data, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_windV, tmp_data, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_windW, tmp_data, num_bytes, cudaMemcpyHostToDevice);

    free(tmp_data);
}


void cleanup()
{
    // cleanup particles
    cudaFree(dev_particles);

    // cleanup grid
    cudaUnbindTexture(texWindU);
    cudaUnbindTexture(texWindV);
    cudaUnbindTexture(texWindW);
    cudaFree(dev_windU);
    cudaFree(dev_windV);
    cudaFree(dev_windW);
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    WindData wd = readASCIIDataFile(argv[1]);
    printWindData(wd);

    // init settings/configuration
    // init meteorology grid

    //init_grid(256, 256, 16, 32);
    //init_particles(10, 1, 2, 3);

    //print_dev_particles(10);
    //advect_particles<<<1, 16>>>(dev_particles, 10, 256, 256, 16);
    //print_dev_particles(10);
    //advect_particles<<<1, 16>>>(dev_particles, 10, 256, 256, 16);
    //print_dev_particles(10);
    //advect_particles<<<1, 16>>>(dev_particles, 10, 256, 256, 16);
    //print_dev_particles(10);

    //Grid g = Grid(512, 512, 128);
    //for (int t=0; t<10; t++) {
    //    advect_particle(p);
    //    std::cout << p.toString() << std::endl;
    //}

    //cleanup();

//    for (int timestep_i=0; timestep_i < num_timesteps; timestep_i++) {
//        // introduce new particles into the system
//
//        for (int pollutant_i=0; pollutant_i < num_pollutants; pollutant_i++) {
//            // skip terminated particles
//
//            // advect particle
//
//            // determine terrain height
//
//            // dry, wet, decay routines
//            // surface water transport
//        }
//    }

    return 0;
}

