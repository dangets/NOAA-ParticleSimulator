#ifndef WINDDATA_CUH
#define WINDDATA_CUH

#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include "Particles.cuh"


struct HostWindData {
    HostWindData(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        u(num_cells), v(num_cells), w(num_cells)
    { }

    const std::size_t num_x;
    const std::size_t num_y;
    const std::size_t num_z;
    const std::size_t num_t;
    const std::size_t num_cells;

    thrust::host_vector<float> u;
    thrust::host_vector<float> v;
    thrust::host_vector<float> w;

    template <typename WindData>
    HostWindData& operator=(const WindData &other) {
        u = other.u;
        v = other.v;
        w = other.w;

        return *this;
    }
};


struct DeviceWindData {
    DeviceWindData(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        u(num_cells), v(num_cells), w(num_cells)
    { }

    template<typename WindData>
    DeviceWindData(WindData &wd) :
        num_x(wd.num_x), num_y(wd.num_y), num_z(wd.num_z), num_t(wd.num_t),
        num_cells(wd.num_cells),
        u(num_cells), v(num_cells), w(num_cells)
    {
        u = wd.u;
        v = wd.v;
        w = wd.w;
    }

    const std::size_t num_x;
    const std::size_t num_y;
    const std::size_t num_z;
    const std::size_t num_t;
    const std::size_t num_cells;

    thrust::device_vector<float> u;
    thrust::device_vector<float> v;
    thrust::device_vector<float> w;

    template <typename WindData>
    DeviceWindData& operator=(const WindData &other) {
        u = other.u;
        v = other.v;
        w = other.w;

        return *this;
    }
};


struct AdvectFunctor {
    template <typename WindData>
    AdvectFunctor(const WindData &wd, const float &t_) :
        u(thrust::raw_pointer_cast(&wd.u[0])),
        v(thrust::raw_pointer_cast(&wd.v[0])),
        w(thrust::raw_pointer_cast(&wd.w[0])),
        t(t_), num_x(wd.num_x), num_y(wd.num_y), num_z(wd.num_z), num_t(wd.num_t)
    { }

    const float *u;
    const float *v;
    const float *w;

    const float t;

    const std::size_t num_x;
    const std::size_t num_y;
    const std::size_t num_z;
    const std::size_t num_t;

    __host__ __device__
    size_t get_index(size_t x, size_t y, size_t z, size_t t) const {
        return x + y * num_x + z * num_y * num_x + t * num_x * num_y * num_z;
    }

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


template <typename Particles, typename WindData>
void advectParticles(Particles &p, WindData &wd, float t)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.end(), p.pos_y.end(), p.pos_z.end())),
        AdvectFunctor(wd, t)
    );
}


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


#endif /* end of include guard: WINDDATA_CUH */
