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

#ifndef WINDDATA_CUH
#define WINDDATA_CUH

#include <vector>
#include <iostream>
#include <fstream>

#include <boost/tr1/memory.hpp>     // for shared_ptr (standard in C++11)
//#include <boost/date_time/posix_time/posix_time.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>


//#include <GeographicLib/GeoCoords.h>
//
//struct WindDataLocation {
//    GeographicLib::GeoCoords coord; // lat, lon
//    float height;                   // m
//    boost::posix_time::ptime t0;
//};
//
//
//struct WindDataSize {
//    // NOTE: can get x, y, z, t spacing from shape and location
//    const float         dx;
//    const float         dy;
//    const float         dz;
//    const time_duration dt;
//};


class WindDataShape {
    public:
        WindDataShape(size_t x_=1, size_t y_=1, size_t z_=1, size_t t_=1)
            : m_x(x_), m_y(y_), m_z(z_), m_t(t_),
              m_size(x_ * y_ * z_ * t_)
        { }

        bool operator==(const WindDataShape &other) const {
            return (m_x == other.m_x &&
                    m_y == other.m_y &&
                    m_z == other.m_z &&
                    m_t == other.m_t);
        }

        bool operator!=(const WindDataShape &other) const {
            return !(*this == other);
        }

        // getter methods
        inline std::size_t x() const { return m_x; }
        inline std::size_t y() const { return m_y; }
        inline std::size_t z() const { return m_z; }
        inline std::size_t t() const { return m_t; }

        inline std::size_t size() const { return m_size; }

    private:
        std::size_t m_x;
        std::size_t m_y;
        std::size_t m_z;
        std::size_t m_t;
        std::size_t m_size;
};

const WindDataShape DEFAULT_SHAPE = WindDataShape(1, 1, 1, 1);



struct WindDataThrustHost {
    WindDataShape shape;

    thrust::host_vector<float> u;
    thrust::host_vector<float> v;
    thrust::host_vector<float> w;

    // constructor
    WindDataThrustHost(const WindDataShape &shape=DEFAULT_SHAPE)
    {
        set_shape(shape);
    }

    template<typename WindData>
    explicit WindDataThrustHost(const WindData &wd)
        : shape(wd.shape),
          u(shape.size()), v(shape.size()), w(shape.size())
    {
        u = wd.u;
        v = wd.v;
        w = wd.w;
    }

    void set_shape(const WindDataShape &shape)
    {
        this->shape = shape;
        u.resize(shape.size());
        v.resize(shape.size());
        w.resize(shape.size());
    }
};


struct WindDataThrustDevice {
    WindDataShape shape;

    thrust::device_vector<float> u;
    thrust::device_vector<float> v;
    thrust::device_vector<float> w;

    // constructor
    WindDataThrustDevice(const WindDataShape &shape=DEFAULT_SHAPE)
    {
        set_shape(shape);
    }

    template<typename WindData>
    explicit WindDataThrustDevice(const WindData &wd)
        : shape(wd.shape),
          u(shape.size()), v(shape.size()), w(shape.size())
    {
        u = wd.u;
        v = wd.v;
        w = wd.w;
    }

    void set_shape(const WindDataShape &shape=DEFAULT_SHAPE)
    {
        this->shape = shape;
        u.resize(shape.size());
        v.resize(shape.size());
        w.resize(shape.size());
    }
};


class WindDataThrustASCIIConverter {
    public:
        static void encode(const WindDataThrustHost &host, std::ostream &out=std::cout);
        static void fill_from_stream(WindDataThrustHost &wind, std::istream &in);
        static WindDataThrustHost from_file(const std::string &fname);
};


//class WindDataVTKConverter {
//    public:
//        static void encode();
//        static WindDataThrustHost decode();
//};




//namespace CUDATextureReferences {
//    /* NOTE: this release of CUDA (v4.0) requires texture references to be GLOBALLY defined
//     *  that is the reason that this exists in a namespace rather than a simple class
//     *
//     *  with the current implementation - only ONE WindDataTextureMemory instance can exist.
//     *  To be able to create more instances (limited number regardless), we need to declare
//     *  more texture references here and probably do template magic with WindDataTextureMemory
//     */
//    texture<float4, 3> vel_curr;
//    texture<float4, 3> vel_next;
//};

texture<float4, 3> vel_curr;
texture<float4, 3> vel_next;


struct WindDataTextureMemoryAccessor {
    //__device__
    //float4 operator()(float x, float y, float z, float t) const {
    //    return make_float4(x, y, z, 0.0f);
    //}

    __device__
    float4 operator()(float x, float y, float z, float t) const {
        // use an offset of 0.5f for "table lookup behavior"
        // see CUDA C Programming Guide Appendix E.3 (v4.2)

        float4 v0 = tex3D(vel_curr, x+0.5f, y+0.5f, z+0.5f);
        //float4 v1 = tex3D(vel_next, x+0.5f, y+0.5f, z+0.5f);

        // TODO: need to interpolate between vel_curr (v0) and
        //  vel_next (v1) based on "t" distance to each
        return v0;
    }
};


struct WindDataTextureMemory {
    // constructor
    WindDataTextureMemory(const WindDataShape &shape);

    // bind the two cuArrays closest to 't' to vel_curr and vel_next texture references
    void set_current_t(const float t) {
        cudaError err;

        // TODO: update cuArrays index based on what 't' is
        err = cudaBindTextureToArray(vel_curr, cuArrays[0].get(), cuChannelDesc);
        if (err != cudaSuccess) {
            std::string str = "error during cudaBindTextureToArray vel_curr: ";
            str += cudaGetErrorString(err);
            throw std::runtime_error(str);
        }

        // TODO: update cuArrays index based on what 't' is
        err = cudaBindTextureToArray(vel_next, cuArrays[0].get(), cuChannelDesc);
        if (err != cudaSuccess) {
            std::string str = "error during cudaBindTextureToArray vel_next: ";
            str += cudaGetErrorString(err);
            throw std::runtime_error(str);
        }
    }


    void set_filter_mode(cudaTextureFilterMode mode) {
        vel_curr.filterMode = mode;
        vel_next.filterMode = mode;
    }

    void set_address_mode(cudaTextureAddressMode mode) {
        vel_curr.addressMode[0] = mode;
        vel_curr.addressMode[1] = mode;
        vel_curr.addressMode[2] = mode;

        vel_next.addressMode[0] = mode;
        vel_next.addressMode[1] = mode;
        vel_next.addressMode[2] = mode;
    }

    WindDataTextureMemoryAccessor get_accessor() {
        return WindDataTextureMemoryAccessor();
    }

    // data members ------------------
    WindDataShape shape;

    cudaExtent cuExtent;
    cudaChannelFormatDesc cuChannelDesc;
    std::vector<std::tr1::shared_ptr<cudaArray> > cuArrays;
};


// copy data from WindDataThrustHost to WindDataTextureMemory
void copy(const WindDataThrustHost &from, WindDataTextureMemory &to);
void copy(const WindDataTextureMemory &from, WindDataThrustHost &to);


#endif /* end of include guard: WINDDATA_CUH */

