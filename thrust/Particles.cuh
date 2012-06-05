#ifndef PARTICLES_CUH
#define PARTICLES_CUH


#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>


struct HostParticles {
    HostParticles(std::size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length) { }

    std::size_t length;

    thrust::host_vector<float> pos_x;
    thrust::host_vector<float> pos_y;
    thrust::host_vector<float> pos_z;

    //std::vector<float> vel_u;
    //std::vector<float> vel_v;
    //std::vector<float> vel_w;

    // type
    // source_id
    // age / birthtime
    // has_deposited
    // ...


    template <typename Particles>
    HostParticles& operator=(const Particles &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        return *this;
    }
};


struct DeviceParticles {
    DeviceParticles(std::size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length) { }

    std::size_t length;

    thrust::device_vector<float> pos_x;
    thrust::device_vector<float> pos_y;
    thrust::device_vector<float> pos_z;

    template <typename Particles>
    DeviceParticles& operator=(const Particles &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        return *this;
    }
};


template <typename Particles>
void ParticlesPrint(Particles &p, std::ostream &out) {
    for (size_t i=0; i<p.length; i++) {
        out << p.pos_x[i] << " "
            << p.pos_y[i] << " "
            << p.pos_z[i] << std::endl;
    }
};



template <typename Particles>
void ParticlesFillSequence(Particles &p) {
    thrust::sequence(p.pos_x.begin(), p.pos_x.end());
    thrust::sequence(p.pos_y.begin(), p.pos_y.end());
    thrust::sequence(p.pos_z.begin(), p.pos_z.end());
}


struct ParticlesRandomizePositionFunctor {
    ParticlesRandomizePositionFunctor(float xmin_, float xmax_, float ymin_, float ymax_, float zmin_, float zmax_) :
        xmin(xmin_), xmax(xmax_),
        ymin(ymin_), ymax(ymax_),
        zmin(zmin_), zmax(zmax_) { }

    const float xmin, xmax;
    const float ymin, ymax;
    const float zmin, zmax;

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple tup) const {
        thrust::get<0>(tup) = xmax * (float)drand48();
        thrust::get<1>(tup) = ymax * (float)drand48();
        thrust::get<2>(tup) = zmax * (float)drand48();
    }
};


void ParticlesRandomizePosition(HostParticles &p,
        float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.end(), p.pos_y.end(), p.pos_z.end())),
        ParticlesRandomizePositionFunctor(xmin, xmax, ymin, ymax, zmin, zmax)
    );
}


#endif /* end of include guard: PARTICLES_CUH */
