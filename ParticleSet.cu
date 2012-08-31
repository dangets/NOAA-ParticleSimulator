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

#include "ParticleSet.cuh"


/// --------------------------------------------------------------
/// ------ create_from_particle_source code ----------------------
/// --------------------------------------------------------------
static std::size_t num_total_particles(const ParticleSource &src)
{
    return (size_t)((src.release_stop - src.release_start) * src.release_rate);
}

// fill particles' birthtime in sequence based on source emission start,stop,rate
template <typename Particles>
static void fill_birthtime(Particles &p, const ParticleSource &src)
{
    // rate = particles per second
    // step = 1 / rate
    float start = (float)src.release_start;
    float step = 1 / src.release_rate;
    thrust::sequence(p.birthtime.begin(), p.birthtime.end(), start, step);
}


ParticleSetThrustHost ParticleSetThrustHost_from_particle_source(const ParticleSource &src)
{
    ParticleSetThrustHost p(num_total_particles(src));
    fill_birthtime(p, src);
    randomize_positions(p,
            src.position.x, src.position.x + src.size.x,
            src.position.y, src.position.y + src.size.y,
            src.position.z, src.position.z + src.size.z);

    return p;
}

ParticleSetThrustDevice ParticleSetThrustDevice_from_particle_source(const ParticleSource &src)
{
    ParticleSetThrustDevice p(num_total_particles(src));
    fill_birthtime(p, src);
    randomize_positions(p,
            src.position.x, src.position.x + src.size.x,
            src.position.y, src.position.y + src.size.y,
            src.position.z, src.position.z + src.size.z);

    return p;
}


/// --------------------------------------------------------------
/// ------ randomize_positions code ------------------------------
/// --------------------------------------------------------------
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

/**
 * functor to randomize particle positions
 */
struct ParticlesRandomizePositionsFunctor {
    ParticlesRandomizePositionsFunctor(float xmin_, float xmax_, float ymin_, float ymax_, float zmin_, float zmax_) :
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


/**
 * template to reduce code duplication between the
 * ParticleSetThrustHost and ParticleSetThrustDevice functions
 */
template <typename Particles>
static void randomize_positions_generic(Particles &p,
                                float xmin, float xmax,
                                float ymin, float ymax,
                                float zmin, float zmax)
{
    thrust::counting_iterator<int> id_begin(0);
    thrust::counting_iterator<int> id_end(p.size());

    thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(id_begin, p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(id_end,   p.pos_x.end(),   p.pos_y.end(),   p.pos_z.end())),
            ParticlesRandomizePositionsFunctor(xmin, xmax, ymin, ymax, zmin, zmax)
    );
}


void randomize_positions(ParticleSetThrustHost &p,
                        float xmin, float xmax,
                        float ymin, float ymax,
                        float zmin, float zmax)
{
    randomize_positions_generic(p, xmin, xmax, ymin, ymax, zmin, zmax);
}

void randomize_positions(ParticleSetThrustDevice &p,
                        float xmin, float xmax,
                        float ymin, float ymax,
                        float zmin, float zmax)
{
    randomize_positions_generic(p, xmin, xmax, ymin, ymax, zmin, zmax);
}



/// --------------------------------------------------------------
/// ------ fill_positions code -----------------------------------
/// --------------------------------------------------------------
struct ParticlesFillPositionsFunctor {
    ParticlesFillPositionsFunctor(float x_, float y_, float z_)
        : x(x_), y(y_), z(z_) { }

    const float x;
    const float y;
    const float z;

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple tup) const {
        thrust::get<0>(tup) = x;
        thrust::get<1>(tup) = y;
        thrust::get<2>(tup) = z;
    }
};


template <typename Particles>
static void fill_positions_generic(Particles &p, float x, float y, float z)
{
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.begin(), p.pos_y.begin(), p.pos_z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(p.pos_x.end(),   p.pos_y.end(),   p.pos_z.end())),
        ParticlesFillPositionsFunctor(x, y, z)
    );
}

void fill_positions(ParticleSetThrustHost &p, float x, float y, float z)
{
    fill_positions_generic(p, x, y, z);
}

void fill_positions(ParticleSetThrustDevice &p, float x, float y, float z)
{
    fill_positions_generic(p, x, y, z);
}

