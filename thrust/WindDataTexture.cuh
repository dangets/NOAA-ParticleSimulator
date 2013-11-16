#ifndef WINDDATATEXTURE_CUH
#define WINDDATATEXTURE_CUH

#include <iostream>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>

#include "WindData.cuh"
#include "Particles.cuh"


namespace WindDataTexture {
    // CUDA texture references
    texture<float4, 3> tex_vel_curr;
    texture<float4, 3> tex_vel_next;

    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;

    cudaArray **cuArrays;      // array of valid length num_t
    cudaExtent cuExtent;
    cudaChannelFormatDesc cuChannelDesc;

    // expects num_x, num_y, num_z, and cuExtent to be set before calling...
    void allocate_cuArrays() {
        cudaError err;

        // TODO: free cuArrays if already previously malloc'd
        cuArrays = (cudaArray **)malloc(sizeof(cudaArray *) * num_t);
        for (size_t i=0; i<num_t; ++i) {
            err = cudaMalloc3DArray(&(cuArrays[i]), &cuChannelDesc, cuExtent);
            if (err != cudaSuccess) {
                throw std::runtime_error("error during cudaMalloc3DArray");
            }
        }
    }

    void cleanup() {
        cudaError err;

        for (size_t i=0; i<num_t; ++i) {
            err = cudaFreeArray(cuArrays[i]);
            if (err != cudaSuccess) {
                throw std::runtime_error("error during cudaFreeArray");
            }
        }
    }

    // texture copying
    struct WindDataCopyFunctor {
        template <typename Tuple>
        __host__
        void operator()(Tuple tup) const
        {
            float4 &d = thrust::get<0>(tup);
            const float &u = thrust::get<1>(tup);
            const float &v = thrust::get<2>(tup);
            const float &w = thrust::get<3>(tup);

            d.x = u;
            d.y = v;
            d.z = w;
        }
    };

    template <typename WindData>
    void copy_wind_data_to_textures(const WindData &wind) {
        cudaError err;

        num_x = wind.num_x;
        num_y = wind.num_y;
        num_z = wind.num_z;
        num_t = wind.num_t;

        cuExtent = make_cudaExtent(num_x, num_y, num_z);
        cuChannelDesc = cudaCreateChannelDesc<float4>();

        allocate_cuArrays();

        // copy wind fields to float4 arrays
        size_t xyz_size = num_x * num_y * num_z;
        thrust::host_vector<float4> tmp(xyz_size);
        for (size_t t=0; t<num_t; ++t) {
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                        tmp.begin(),
                        wind.u.begin()+t*xyz_size,
                        wind.v.begin()+t*xyz_size,
                        wind.w.begin()+t*xyz_size)),
                thrust::make_zip_iterator(thrust::make_tuple(
                        tmp.end(),
                        wind.u.begin()+((t+1)*xyz_size),
                        wind.v.begin()+((t+1)*xyz_size),
                        wind.w.begin()+((t+1)*xyz_size))),
                WindDataCopyFunctor()
            );

            // copy from tmp vector into the cuArray
            cudaMemcpy3DParms copyParms = {0};
            copyParms.srcPtr = make_cudaPitchedPtr((void *)(&(tmp[0])), num_x * sizeof(float4), num_x, num_y);
            copyParms.dstArray = cuArrays[t];
            copyParms.extent = cuExtent;
            copyParms.kind = cudaMemcpyHostToDevice;

            err = cudaMemcpy3D(&copyParms);
            if (err != cudaSuccess) {
                std::string err_msg = "error during cudaMemcpy3D: ";
                err_msg += cudaGetErrorString(err);
                throw std::runtime_error(err_msg);
            }
        }
    }


    // texture address mode
    void set_texture_address_mode(cudaTextureAddressMode mode) {
        tex_vel_curr.filterMode = cudaFilterModeLinear;
        tex_vel_curr.addressMode[0] = mode;
        tex_vel_curr.addressMode[1] = mode;
        tex_vel_curr.addressMode[2] = mode;

        tex_vel_next.filterMode = cudaFilterModeLinear;
        tex_vel_next.addressMode[0] = mode;
        tex_vel_next.addressMode[1] = mode;
        tex_vel_next.addressMode[2] = mode;
    }


    struct AdvectFunctor {
        AdvectFunctor(const time_t &t_) :
            t(t_)
        { }

        const time_t t;

        template <typename Tuple>
        __device__
        void operator()(Tuple tup) const
        {
            float &x = thrust::get<0>(tup);
            float &y = thrust::get<1>(tup);
            float &z = thrust::get<2>(tup);
            time_t &birthtime = thrust::get<3>(tup);
            bool &has_deposited = thrust::get<4>(tup);

            if (birthtime > t)
                return;
            if (has_deposited)
                return;

            // TODO: need to add the 0.5 for "table-lookup" behavior
            float4 vel_0 = tex3D(tex_vel_curr, x+0.5f, y+0.5f, z+0.5f);

            // first guess position P1
            float x1 = x + vel_0.x;
            float y1 = y + vel_0.y;
            float z1 = z + vel_0.z;

            float4 vel_1 = tex3D(tex_vel_next, x1+0.5f, y1+0.5f, z1+0.5f);

            x += 0.5f * vel_0.x + 0.5f * vel_1.x;
            y += 0.5f * vel_0.y + 0.5f * vel_1.y;
            z += 0.5f * vel_0.z + 0.5f * vel_1.z;
        }
    };


    void advectParticles(DeviceParticles &p, time_t t)
    {
        cudaError err;

        // TODO: update tex_vel_curr and tex_vel_next bindings based on 't'
        err = cudaBindTextureToArray(tex_vel_curr, cuArrays[0], cuChannelDesc);
        if (err != cudaSuccess) {
            throw std::runtime_error("error during cudaBindTextureToArray(tex_vel_curr, ...");
        }
        err = cudaBindTextureToArray(tex_vel_next, cuArrays[0], cuChannelDesc);
        if (err != cudaSuccess) {
            throw std::runtime_error("error during cudaBindTextureToArray(tex_vel_next, ...");
        }

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
            AdvectFunctor(t)
        );
    }


    struct AdvectRungeKuttaFunctor {
        AdvectRungeKuttaFunctor(const time_t &t_) :
            t(t_)
        { }

        const time_t t;

        template <typename Tuple>
        __device__
        void operator()(Tuple tup) const
        {
            float &x = thrust::get<0>(tup);
            float &y = thrust::get<1>(tup);
            float &z = thrust::get<2>(tup);
            time_t &birthtime = thrust::get<3>(tup);
            bool &has_deposited = thrust::get<4>(tup);

            if (birthtime > t)
                return;
            if (has_deposited)
                return;

            float3 tmp;

            float4 k1 = tex3D(tex_vel_curr, x+0.5f, y+0.5f, z+0.5f);

            tmp.x = x + 0.5f*k1.x;
            tmp.y = y + 0.5f*k1.y;
            tmp.z = z + 0.5f*k1.z;

            float4 k2 = tex3D(tex_vel_curr, tmp.x+0.5f, tmp.y+0.5f, tmp.z+0.5f);

            tmp.x = x + 0.5f*k2.x;
            tmp.y = y + 0.5f*k2.y;
            tmp.z = z + 0.5f*k2.z;

            float4 k3 = tex3D(tex_vel_curr, tmp.x+0.5f, tmp.y+0.5f, tmp.z+0.5f);

            float4 k4 = tex3D(tex_vel_curr, x + k3.x, y + k3.y, z + k3.z);

            x = x + 1.0f/6.0f * (k1.x + k4.x) + 1.0f/3.0f * (k2.x + k3.x);
            y = y + 1.0f/6.0f * (k1.y + k4.y) + 1.0f/3.0f * (k2.y + k3.y);
            z = z + 1.0f/6.0f * (k1.z + k4.z) + 1.0f/3.0f * (k2.z + k3.z);
        }
    };

    void advectParticlesRungeKutta(DeviceParticles &p, time_t t)
    {
        cudaError err;

        // TODO: update tex_vel_curr and tex_vel_next bindings based on 't'
        err = cudaBindTextureToArray(tex_vel_curr, cuArrays[0], cuChannelDesc);
        if (err != cudaSuccess) {
            throw std::runtime_error("error during cudaBindTextureToArray(tex_vel_curr, ...");
        }
        err = cudaBindTextureToArray(tex_vel_next, cuArrays[0], cuChannelDesc);
        if (err != cudaSuccess) {
            throw std::runtime_error("error during cudaBindTextureToArray(tex_vel_next, ...");
        }

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
            AdvectRungeKuttaFunctor(t)
        );
    }
};


// TODO: needs to be a singleton for sole access to WindDataTexture namespace
//      (due to static global declaration requirement of cuda texture references)
struct TextureParticleSimulation {
    TextureParticleSimulation(const HostWindData &wind, time_t cur_step=0) :
        cur_step(cur_step)
    {
        WindDataTexture::copy_wind_data_to_textures(wind);
        WindDataTexture::set_texture_address_mode(cudaAddressModeClamp);
    }

    DeviceParticles *particles;
    time_t cur_step;

    void step() {
        //WindDataTexture::advectParticles(*particles, cur_step);
        WindDataTexture::advectParticlesRungeKutta(*particles, cur_step);
        ++cur_step;
    }
};


#endif /* end of include guard: WINDDATATEXTURE_CUH */
