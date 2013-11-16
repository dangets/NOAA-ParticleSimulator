/*
   Author: Danny George
   High Performance Simulation Laboratory
   Boise State University
 
   Permission is hereby granted, free of charge, to any person obtaining a copy of
   this software and associated documentation files (the "Software"), to deal in
   the Software without restriction, including without limitation the rights to
   use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is furnished to do
   so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#include "advect_runge_kutta.cuh"


namespace {
}

struct AdvectRungeKuttaFunctor {
    template <typename WindData>
    AdvectRungeKuttaFunctor(const WindData &wd, const int &t_)
        : u(thrust::raw_pointer_cast(&wd.u[0])),
            v(thrust::raw_pointer_cast(&wd.v[0])),
            w(thrust::raw_pointer_cast(&wd.w[0])),
            t(t_),
            num_x(wd.shape.x()), num_y(wd.shape.y()), num_z(wd.shape.z()), num_t(wd.shape.t())
    { }

    const float *u;         // pointer to WindData's 'u' data
    const float *v;         // pointer to WindData's 'v' data
    const float *w;         // pointer to WindData's 'w' data

    const int t;

    const size_t num_x;
    const size_t num_y;
    const size_t num_z;
    const size_t num_t;


    __host__ __device__
    inline size_t get_index(size_t x, size_t y, size_t z, size_t t) const {
        // TODO: hardcoded t to 0 for development ~v~~~~~~~~~
        return x + y * num_x + z * num_y * num_x + 0 * num_x * num_y * num_z;
    }

    __host__ __device__
    float3 get_velocity_border0(float x, float y, float z, float t) const {
        float3 ret = make_float3(0.0f, 0.0f, 0.0f);
        size_t x0, x1;
        size_t y0, y1;
        size_t z0, z1;

        // border logic ----------
        if (x <= 0) {
            return ret;
        } else if (x >= num_x-1) {
            return ret;
        } else {
            x0 = (size_t)x;
            x1 = x+1;
        }

        if (y <= 0) {
            return ret;
        } else if (y >= num_y-1) {
            return ret;
        } else {
            y0 = (size_t)y;
            y1 = y+1;
        }

        if (z <= 0) {
            return ret;
        } else if (z >= num_z-1) {
            return ret;
        } else {
            z0 = (size_t)z;
            z1 = z+1;
        }

        // distance from actual point to sampled point index
        float x_d = x - x0;
        float y_d = y - y0;
        float z_d = z - z0;

        size_t i000 = get_index(x0, y0, z0, t);
        size_t i100 = get_index(x1, y1, z0, t);
        size_t i010 = get_index(x0, y1, z0, t);
        size_t i110 = get_index(x1, y1, z0, t);
        size_t i001 = get_index(x0, y0, z1, t);
        size_t i101 = get_index(x1, y0, z1, t);
        size_t i011 = get_index(x0, y1, z1, t);
        size_t i111 = get_index(x1, y1, z1, t);

        float c00 = u[i000] * (1 - x_d) + u[i100] * x_d;
        float c10 = u[i010] * (1 - x_d) + u[i110] * x_d;
        float c01 = u[i001] * (1 - x_d) + u[i101] * x_d;
        float c11 = u[i011] * (1 - x_d) + u[i111] * x_d;

        float c0 = c00 * (1 - y_d) + c10 * y_d;
        float c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.x = (c0 * (1 - z_d) + c1 * z_d);

        c00 = v[i000] * (1 - x_d) + v[i100] * x_d;
        c10 = v[i010] * (1 - x_d) + v[i110] * x_d;
        c01 = v[i001] * (1 - x_d) + v[i101] * x_d;
        c11 = v[i011] * (1 - x_d) + v[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.y = (c0 * (1 - z_d) + c1 * z_d);

        c00 = w[i000] * (1 - x_d) + w[i100] * x_d;
        c10 = w[i010] * (1 - x_d) + w[i110] * x_d;
        c01 = w[i001] * (1 - x_d) + w[i101] * x_d;
        c11 = w[i011] * (1 - x_d) + w[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.z = (c0 * (1 - z_d) + c1 * z_d);

        return ret;
    }


    __host__ __device__
    float3 get_velocity_clamp(float x, float y, float z, float t) const {
        float3 ret;
        size_t x0, x1;
        size_t y0, y1;
        size_t z0, z1;

        // clamping logic ----------
        if (x <= 0) {
            x0 = 0;
            x1 = 0;
        } else if (x >= num_x-1) {
            x0 = num_x-1;
            x1 = num_x-1;
        } else {
            x0 = (size_t)x;
            x1 = x+1;
        }

        if (y <= 0) {
            y0 = 0;
            y1 = 0;
        } else if (y >= num_y-1) {
            y0 = num_y-1;
            y1 = num_y-1;
        } else {
            y0 = (size_t)y;
            y1 = y+1;
        }

        if (z <= 0) {
            z0 = 0;
            z1 = 0;
        } else if (z >= num_z-1) {
            z0 = num_z-1;
            z1 = num_z-1;
        } else {
            z0 = (size_t)z;
            z1 = z+1;
        }

        // distance from actual point to sampled point index
        float x_d = x - x0;
        float y_d = y - y0;
        float z_d = z - z0;

        size_t i000 = get_index(x0, y0, z0, t);
        size_t i100 = get_index(x1, y1, z0, t);
        size_t i010 = get_index(x0, y1, z0, t);
        size_t i110 = get_index(x1, y1, z0, t);
        size_t i001 = get_index(x0, y0, z1, t);
        size_t i101 = get_index(x1, y0, z1, t);
        size_t i011 = get_index(x0, y1, z1, t);
        size_t i111 = get_index(x1, y1, z1, t);

        float c00 = u[i000] * (1 - x_d) + u[i100] * x_d;
        float c10 = u[i010] * (1 - x_d) + u[i110] * x_d;
        float c01 = u[i001] * (1 - x_d) + u[i101] * x_d;
        float c11 = u[i011] * (1 - x_d) + u[i111] * x_d;

        float c0 = c00 * (1 - y_d) + c10 * y_d;
        float c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.x = (c0 * (1 - z_d) + c1 * z_d);

        c00 = v[i000] * (1 - x_d) + v[i100] * x_d;
        c10 = v[i010] * (1 - x_d) + v[i110] * x_d;
        c01 = v[i001] * (1 - x_d) + v[i101] * x_d;
        c11 = v[i011] * (1 - x_d) + v[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.y = (c0 * (1 - z_d) + c1 * z_d);

        c00 = w[i000] * (1 - x_d) + w[i100] * x_d;
        c10 = w[i010] * (1 - x_d) + w[i110] * x_d;
        c01 = w[i001] * (1 - x_d) + w[i101] * x_d;
        c11 = w[i011] * (1 - x_d) + w[i111] * x_d;

        c0 = c00 * (1 - y_d) + c10 * y_d;
        c1 = c01 * (1 - y_d) + c11 * y_d;

        ret.z = (c0 * (1 - z_d) + c1 * z_d);

        return ret;
    }


    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple tup) const
    {
        float &x = thrust::get<0>(tup);
        float &y = thrust::get<1>(tup);
        float &z = thrust::get<2>(tup);
        int   &birthtime = thrust::get<3>(tup);
        bool  &has_deposited = thrust::get<4>(tup);

        if (birthtime > t)
            return;
        if (has_deposited)
            return;

        float3 vel_0 = get_velocity_clamp(x, y, z, t);

        // first guess position P1
        float x1 = x + vel_0.x;
        float y1 = y + vel_0.y;
        float z1 = z + vel_0.z;
        float3 vel_1 = get_velocity_clamp(x1, y1, z1, t+1);

        x += 0.5f * vel_0.x + 0.5f * vel_1.x;
        y += 0.5f * vel_0.y + 0.5f * vel_1.y;
        z += 0.5f * vel_0.z + 0.5f * vel_1.z;
    }
};


template <typename Particles, typename WindData>
void advect_runge_kutta_generic(Particles &p, WindData &wd, float t)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            p.pos_x.begin(),
            p.pos_y.begin(), 
            p.pos_z.begin(),
            p.birthtime.begin(),
            p.has_deposited.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            p.pos_x.end(),
            p.pos_y.end(),
            p.pos_z.end(),
            p.birthtime.end(),
            p.has_deposited.end())),
        AdvectRungeKuttaFunctor(wd, t)
    );
}

void advect_runge_kutta(ParticleSetThrustHost &particles, const WindDataThrustHost &wind, float t)
{
    advect_runge_kutta_generic(particles, wind, t);
}

void advect_runge_kutta(ParticleSetThrustDevice &particles, const WindDataThrustDevice &wind, float t)
{
    advect_runge_kutta_generic(particles, wind, t);
}



struct WindTextureAdvectRungeKuttaFunctor {
    WindTextureAdvectRungeKuttaFunctor(WindDataTextureMemoryAccessor accessor, float t_)
        : t(t_), get_wind(accessor)
    { }

    const float t;
    WindDataTextureMemoryAccessor get_wind;

    template <typename Tuple>
    __device__
    void operator()(Tuple tup) const
    {
        float &x             = thrust::get<0>(tup);
        float &y             = thrust::get<1>(tup);
        float &z             = thrust::get<2>(tup);
        int   &birthtime     = thrust::get<3>(tup);
        bool  &has_deposited = thrust::get<4>(tup);

        if (birthtime > t)
            return;
        if (has_deposited)
            return;

        float3 tmp;
        //tmp.w = t;      // TODO: need to adjust between each 'k' value?

        float4 k1 = get_wind(x, y, z, t);

        tmp.x = x + 0.5f*k1.x;
        tmp.y = y + 0.5f*k1.y;
        tmp.z = z + 0.5f*k1.z;

        float4 k2 = get_wind(tmp.x, tmp.y, tmp.z, t);

        tmp.x = x + 0.5f*k2.x;
        tmp.y = y + 0.5f*k2.y;
        tmp.z = z + 0.5f*k2.z;

        float4 k3 = get_wind(tmp.x, tmp.y, tmp.z, t);

        // TODO: double check k4 equation...
        float4 k4 = get_wind(x+k3.x, y+k3.y, z+k3.z, t);

        x = x + 1.0f/6.0f * (k1.x + k4.x) + 1.0f/3.0f * (k2.x + k3.x);
        y = y + 1.0f/6.0f * (k1.y + k4.y) + 1.0f/3.0f * (k2.y + k3.y);
        z = z + 1.0f/6.0f * (k1.z + k4.z) + 1.0f/3.0f * (k2.z + k3.z);
    }
};



void advect_runge_kutta(ParticleSetThrustDevice &particles, WindDataTextureMemory &wind, float t)
{
    wind.set_current_t(t);

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            particles.pos_x.begin(),
            particles.pos_y.begin(), 
            particles.pos_z.begin(),
            particles.birthtime.begin(),
            particles.has_deposited.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            particles.pos_x.end(),
            particles.pos_y.end(),
            particles.pos_z.end(),
            particles.birthtime.end(),
            particles.has_deposited.end())),
        WindTextureAdvectRungeKuttaFunctor(wind.get_accessor(), t)
    );
}

