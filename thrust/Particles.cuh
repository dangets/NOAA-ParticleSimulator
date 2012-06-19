#ifndef PARTICLES_CUH
#define PARTICLES_CUH


#include <iostream>
#include <cstdlib>
#include <ctime>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>

using std::size_t;
using std::time_t;

struct HostParticles {
    // constructor
    HostParticles(size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length),
        birthtime(length, 0), has_deposited(length, false)
    { }

    // constant across the group variables ---
    size_t length;
    // type
    // source_id

    // individual particle variables ---------
    thrust::host_vector<float> pos_x;
    thrust::host_vector<float> pos_y;
    thrust::host_vector<float> pos_z;
    //Vector vel_u;
    //Vector vel_v;
    //Vector vel_w;
    thrust::host_vector<time_t> birthtime;
    thrust::host_vector<bool> has_deposited;
    // ...

    // convenience functions -----------------
    template <typename TOther>
    HostParticles& operator=(const TOther &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        birthtime = other.birthtime;
        has_deposited = other.has_deposited;

        return *this;
    }
};


struct DeviceParticles {
    // constructor
    DeviceParticles(size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length),
        birthtime(length), has_deposited(length)
    { }

    // constant across the group variables ---
    size_t length;
    // type
    // source_id

    // individual particle variables ---------
    thrust::device_vector<float> pos_x;
    thrust::device_vector<float> pos_y;
    thrust::device_vector<float> pos_z;
    //Vector vel_u;
    //Vector vel_v;
    //Vector vel_w;
    thrust::device_vector<time_t> birthtime;
    thrust::device_vector<bool> has_deposited;
    // ...

    // convenience functions -----------------
    template <typename TOther>
    DeviceParticles& operator=(const TOther &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        birthtime = other.birthtime;
        has_deposited = other.has_deposited;

        return *this;
    }
};



struct OpenGLParticleCopy {
    template <typename Tuple>
    __host__ __device__
    float3 operator()(Tuple tup) const {
        return make_float3(
                thrust::get<0>(tup),
                thrust::get<1>(tup),
                thrust::get<2>(tup)
        );
    }
};


struct OpenGLParticles {
    size_t length;
    GLuint pos_vbo;
    cudaGraphicsResource *pos_cvbo;

    OpenGLParticles(size_t length_) :
        length(length_)
    {
        size_t pos_size = length * sizeof(float4);

        // create and allocate OpenGL buffers
        glGenBuffers(1, &pos_vbo);

        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glBufferData(GL_ARRAY_BUFFER, pos_size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLenum gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) {
            throw std::runtime_error("error initializing OpenGL VBO");
        }

        // register with CUDA
        cudaGraphicsGLRegisterBuffer(&pos_cvbo, pos_vbo, cudaGraphicsMapFlagsWriteDiscard);
    }

    ~OpenGLParticles()
    {
        // unregister this buffer object with CUDA
        cudaGraphicsUnregisterResource(pos_cvbo);
        glDeleteBuffers(1, &pos_vbo);
    }

    void copy(const DeviceParticles &p) {
        float3 *raw_ptr;
        size_t buf_size;

        mapCUDA();
        cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, pos_cvbo);
        thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(raw_ptr);

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.end(), p.pos_y.end(), p.pos_z.end())),
            dev_ptr,
            OpenGLParticleCopy()
        );

        unmapCUDA();
    }

    void mapCUDA() {
        cudaGraphicsMapResources(1, &pos_cvbo, 0);
    }

    void unmapCUDA() {
        cudaGraphicsUnmapResources(1, &pos_cvbo, 0);
    }
};


/// ------ Debugging utils ------------------
template <typename Particles>
inline static void ParticlePrint(const Particles &p, size_t i, std::ostream &out) {
    out << p.pos_x[i] << " "
        << p.pos_y[i] << " "
        << p.pos_z[i] << " "
        //<< p.birthtime[i] << " "
        //<< p.has_deposited[i] << " "
        << std::endl;
}


template <typename Particles>
void ParticlesPrint(const Particles &p, std::ostream &out) {
    for (size_t i=0; i<p.length; i++) {
        ParticlePrint(p, i, out);
    }
}

template <typename Particles>
void ParticlesPrintActive(const Particles &p, std::ostream &out, time_t t) {
    for (size_t i=0; i<p.length; i++) {
        if (p.birthtime[i] > t)
            continue;

        ParticlePrint(p, i, out);
    }
}




/// ------ Randomize Positions ---------------
__host__ __device__
unsigned int hash(unsigned int a)
{
    // copied from thrust/examples/monte_carlo.cu
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct ParticlesRandomizePositionFunctor {
    ParticlesRandomizePositionFunctor(float xmin_, float xmax_, float ymin_, float ymax_, float zmin_, float zmax_) :
        xmin(xmin_), xmax(xmax_),
        ymin(ymin_), ymax(ymax_),
        zmin(zmin_), zmax(zmax_)
    { }

    const float xmin, xmax;
    const float ymin, ymax;
    const float zmin, zmax;

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple tup) const {
        unsigned int seed = hash(thrust::get<0>(tup));
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u01(0, 1);

        thrust::get<1>(tup) = xmin + (xmax-xmin) * u01(rng);
        thrust::get<2>(tup) = ymin + (ymax-ymin) * u01(rng);
        thrust::get<3>(tup) = zmin + (zmax-zmin) * u01(rng);
    }
};


template <typename Particles>
void ParticlesRandomizePosition(
        Particles &p,
        float xmin, float xmax,
        float ymin, float ymax,
        float zmin, float zmax)
{
    thrust::counting_iterator<int> id_begin(0);
    thrust::counting_iterator<int> id_end(p.length);

    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(id_begin, p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(id_end, p.pos_x.end(), p.pos_y.end(), p.pos_z.end())),
            ParticlesRandomizePositionFunctor(xmin, xmax, ymin, ymax, zmin, zmax)
    );
}



/// ------ Fill positions with a constant ----
template <typename Particles>
void ParticlesFillPosition(Particles &p, float x, float y, float z)
{
    // TODO: this would probably be faster with a single functor
    thrust::fill(p.pos_x.begin(), p.pos_x.end(), x);
    thrust::fill(p.pos_y.begin(), p.pos_y.end(), y);
    thrust::fill(p.pos_z.begin(), p.pos_z.end(), z);
}


/// ------ Fill births based on start/stop/rate vars
template <typename Particles>
void ParticlesFillBirthTime(Particles &p, time_t start, time_t stop, float rate)
{
    // rate = p / s
    // step = s / p
    float step = start + 1 / rate;
    thrust::sequence(p.birthtime.begin(), p.birthtime.end(), (float)start, step);
}



#endif /* end of include guard: PARTICLES_CUH */
