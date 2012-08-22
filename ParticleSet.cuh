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

#ifndef PARTICLESET_CUH
#define PARTICLESET_CUH

#include "ParticleSource.hpp"

#include <cstring>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>



/// --------------------------------------------------------------
/// ------ ParticleSet thrust classes ----------------------------
/// --------------------------------------------------------------
struct ParticleSetThrustHost {
    thrust::host_vector<float> pos_x;
    thrust::host_vector<float> pos_y;
    thrust::host_vector<float> pos_z;

    thrust::host_vector<int>   birthtime;
    thrust::host_vector<bool>  has_deposited;

    // constructor
    ParticleSetThrustHost(std::size_t initial_size)
        : pos_x(initial_size), pos_y(initial_size), pos_z(initial_size),
          birthtime(initial_size), has_deposited(initial_size)
    { }

    template <typename TOther>
    ParticleSetThrustHost& operator=(const TOther &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        birthtime     = other.birthtime;
        has_deposited = other.has_deposited;

        return *this;
    }

    inline std::size_t size() const {
        return pos_x.size();
    }
};


struct ParticleSetThrustDevice {
    thrust::device_vector<float> pos_x;
    thrust::device_vector<float> pos_y;
    thrust::device_vector<float> pos_z;

    thrust::device_vector<int>   birthtime;
    thrust::device_vector<bool>  has_deposited;

    // constructor
    ParticleSetThrustDevice(std::size_t initial_size)
        : pos_x(initial_size), pos_y(initial_size), pos_z(initial_size),
          birthtime(initial_size), has_deposited(initial_size)
    { }

    template <typename RHS>
    ParticleSetThrustDevice& operator=(const RHS &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        birthtime     = other.birthtime;
        has_deposited = other.has_deposited;

        return *this;
    }

    inline std::size_t size() const {
        return pos_x.size();
    }
};


ParticleSetThrustHost   ParticleSetThrustHost_from_particle_source(const ParticleSource &src);
ParticleSetThrustDevice ParticleSetThrustDevice_from_particle_source(const ParticleSource &src);


/// ------ Randomize Positions ---------------
void randomize_positions(ParticleSetThrustHost &p,
                        float xmin, float xmax,
                        float ymin, float ymax,
                        float zmin, float zmax);

void randomize_positions(ParticleSetThrustDevice &p,
                        float xmin, float xmax,
                        float ymin, float ymax,
                        float zmin, float zmax);


/// ------ Fill positions with a constant ----
void fill_positions(ParticleSetThrustHost   &p, float x, float y, float z);
void fill_positions(ParticleSetThrustDevice &p, float x, float y, float z);


/// ------ Fill births based on start/stop/rate vars
//template <typename Particles>
//void ParticlesFillBirthTime(Particles &p, time_t start, time_t stop, float rate)
//{
//    // rate = p / s
//    // step = s / p
//    float step = start + 1 / rate;
//    thrust::sequence(p.birthtime.begin(), p.birthtime.end(), (float)start, step);
//}



#endif /* end of include guard: PARTICLESET_CUH */
