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

#include "WindData.cuh"

/*************************************************
 * WindDataThrustASCIIConverter
 *************************************************/
void WindDataThrustASCIIConverter::encode(const WindDataThrustHost &wind, std::ostream &out)
{
    out << wind.shape.x() << " "
        << wind.shape.y() << " "
        << wind.shape.z() << " "
        << wind.shape.t() << " "
        << std::endl;

    out << std::endl;

    for (size_t t=0; t<wind.shape.t(); ++t) {
        size_t t_offset = t * wind.shape.z();
        for (size_t z=0; z<wind.shape.z(); ++z) {
            size_t z_offset = z * wind.shape.y();
            for (size_t y=0; y<wind.shape.y(); ++y) {
                size_t y_offset = y * wind.shape.x();
                for (size_t x=0; x<wind.shape.x(); ++x) {
                    size_t i = x + y_offset + z_offset + t_offset;
                    out << wind.u[i] << " "
                        << wind.v[i] << " "
                        << wind.w[i] << " ";
                }
                out << std::endl;
            }
            out << std::endl;
        }
        out << std::endl;
    }
}


void WindDataThrustASCIIConverter::fill_from_stream(WindDataThrustHost &wind, std::istream &in)
{
    size_t num_x;
    size_t num_y;
    size_t num_z;
    size_t num_t;

    in >> num_x;
    in >> num_y;
    in >> num_z;
    in >> num_t;

    WindDataShape shape(num_x, num_y, num_z, num_t);

    wind.set_shape(shape);

    for (size_t t=0; t<num_t; t++) {
        size_t t_offset = t * num_z * num_y * num_x;
        for (size_t z=0; z<num_z; z++) {
            size_t z_offset = z * num_y * num_x;
            for (size_t y=0; y<num_y; y++) {
                size_t y_offset = y * num_x;
                for (size_t x=0; x<num_x; x++) {
                    size_t i = x + y_offset + z_offset + t_offset;

                    in >> wind.u[i];
                    in >> wind.v[i];
                    in >> wind.w[i];
                }
            }
        }
    }
}


WindDataThrustHost WindDataThrustASCIIConverter::from_file(const std::string &fname)
{
    std::ifstream in;

    in.open(fname.c_str());
    WindDataThrustHost wind;
    fill_from_stream(wind, in);
    in.close();

    return wind;
}





/*************************************************
 * WindDataTextureMemory
 *************************************************/
/**
 * Functor to provide an appropriate cleanup for a cudaArray
 * this is called when using the automatic cleanup below
 */
#include <iostream>
struct CudaArrayDeleteFunctor {
    void operator()(cudaArray *ptr) {
        //std::cerr << "CudaArrayDeleteFunctor called" << std::endl;
        cudaFreeArray(ptr);
    }
};


// constructor
WindDataTextureMemory::WindDataTextureMemory(const WindDataShape &shape)
    : shape(shape)
{
    cuExtent = make_cudaExtent(shape.x(), shape.y(), shape.z());
    cuChannelDesc = cudaCreateChannelDesc<float4>();

    // initialize the vector of cudaArrays -
    //  using shared_ptr for reference counted automatic cleanup
    for (std::size_t i=0; i<shape.t(); ++i) {
        cudaError err;
        cudaArray *ptr;
        err = cudaMalloc3DArray(&ptr, &cuChannelDesc, cuExtent);
        if (err != cudaSuccess) {
            std::string err_msg = "error during cudaMalloc3DArray: ";
            err_msg += cudaGetErrorString(err);
            throw std::runtime_error(err_msg);
        }
        cuArrays.push_back(std::tr1::shared_ptr<cudaArray>(ptr, CudaArrayDeleteFunctor()));
    }
}


/// --------------------------------------------------------------
/// ------ Copying from thrust wind data objects -----------------
/// --------------------------------------------------------------
struct CopyToTextureFunctor {
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

struct CopyFromTextureFunctor {
    template <typename Tuple>
    __host__
    void operator()(Tuple tup) const
    {
        const float4 &d = thrust::get<0>(tup);
        float &u = thrust::get<1>(tup);
        float &v = thrust::get<2>(tup);
        float &w = thrust::get<3>(tup);

        u = d.x;
        v = d.y;
        w = d.z;
    }
};


void copy(const WindDataThrustHost &h_wind, WindDataTextureMemory &t_wind)
{
    if (h_wind.shape != t_wind.shape) {
        throw std::runtime_error("h_wind and t_wind shapes don't match!");
    }

    const size_t &num_x = t_wind.shape.x();
    const size_t &num_y = t_wind.shape.y();
    const size_t &num_z = t_wind.shape.z();
    const size_t &num_t = t_wind.shape.t();

    // copy separate u,v,w wind data to interleaved float4 arrays
    size_t xyz_size = num_x * num_y * num_z;
    thrust::host_vector<float4> tmp(xyz_size);

    for (size_t t=0; t<num_t; ++t) {
        cudaError err;

        // copy separate u,v,w into a temporary float4 vector
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                    tmp.begin(),
                    h_wind.u.begin()+t*xyz_size,
                    h_wind.v.begin()+t*xyz_size,
                    h_wind.w.begin()+t*xyz_size)),
            thrust::make_zip_iterator(thrust::make_tuple(
                    tmp.end(),
                    h_wind.u.begin()+((t+1)*xyz_size),
                    h_wind.v.begin()+((t+1)*xyz_size),
                    h_wind.w.begin()+((t+1)*xyz_size))),
            CopyToTextureFunctor()
        );

        // copy from temporary vector into the cuArray corresponding with 't'
        cudaMemcpy3DParms copyParms = { 0 };
        copyParms.srcPtr = make_cudaPitchedPtr((void *)(&(tmp[0])), num_x * sizeof(float4), num_x, num_y);
        copyParms.dstArray = t_wind.cuArrays[t].get();
        copyParms.extent = t_wind.cuExtent;
        copyParms.kind = cudaMemcpyHostToDevice;

        err = cudaMemcpy3D(&copyParms);
        if (err != cudaSuccess) {
            std::string err_msg = "error during cudaMemcpy3D: ";
            err_msg += cudaGetErrorString(err);
            throw std::runtime_error(err_msg);
        }
    }
}


void copy(const WindDataTextureMemory &t_wind, WindDataThrustHost &h_wind)
{
    h_wind.set_shape(t_wind.shape);

    const size_t &num_x = t_wind.shape.x();
    const size_t &num_y = t_wind.shape.y();
    const size_t &num_z = t_wind.shape.z();
    const size_t &num_t = t_wind.shape.t();

    // copy separate u,v,w wind data to interleaved float4 arrays
    size_t xyz_size = num_x * num_y * num_z;
    thrust::host_vector<float4> tmp(xyz_size);

    for (size_t t=0; t<num_t; ++t) {
        cudaError err;

        // copy from cuArray[t] to temporary float4 vector
        cudaMemcpy3DParms copyParms = { 0 };
        copyParms.srcArray = t_wind.cuArrays[t].get();
        copyParms.dstPtr = make_cudaPitchedPtr((void *)(&(tmp[0])), num_x * sizeof(float4), num_x, num_y);
        copyParms.extent = t_wind.cuExtent;
        copyParms.kind = cudaMemcpyDeviceToHost;

        err = cudaMemcpy3D(&copyParms);
        if (err != cudaSuccess) {
            std::string err_msg = "error during cudaMemcpy3D: ";
            err_msg += cudaGetErrorString(err);
            throw std::runtime_error(err_msg);
        }

        // copy from temporary float4 vector to separate u,v,w
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                    tmp.begin(),
                    h_wind.u.begin()+t*xyz_size,
                    h_wind.v.begin()+t*xyz_size,
                    h_wind.w.begin()+t*xyz_size)),
            thrust::make_zip_iterator(thrust::make_tuple(
                    tmp.end(),
                    h_wind.u.begin()+((t+1)*xyz_size),
                    h_wind.v.begin()+((t+1)*xyz_size),
                    h_wind.w.begin()+((t+1)*xyz_size))),
            CopyFromTextureFunctor()
        );
    }
}

