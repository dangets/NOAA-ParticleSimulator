#ifndef PARTICLES_CUH
#define PARTICLES_CUH


#include <iostream>
#include <cstdlib>

#include <thrust/iterator/zip_iterator.h>

using std::size_t;


// TODO: write big long documentation describing the code below...
template<typename Vector>
struct Particles {
    typedef typename Vector::value_type T;

    // iterator logic ------------------------
    typedef thrust::zip_iterator< 
        thrust::tuple< 
            typename Vector::iterator,  // pos_x
            typename Vector::iterator,  // pos_y
            typename Vector::iterator   // pos_z
        > 
    > iterator;

    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(
                    pos_x.begin(),
                    pos_y.begin(),
                    pos_z.begin()
                ));
    }

    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(
                    pos_x.end(),
                    pos_y.end(),
                    pos_z.end()
                ));
    }

    // reference struct that looks like a single particle struct
    //  this is what functors can accept as an argument
    typedef typename iterator::reference zip_reference;
    struct ref {
        T& pos_x;
        T& pos_y;
        T& pos_z;

        ref(zip_reference r) :
            pos_x(thrust::get<0>(r)),
            pos_y(thrust::get<1>(r)),
            pos_z(thrust::get<2>(r))
        { }
    };

    // constructor
    Particles(size_t length_) :
        length(length_),
        pos_x(length), pos_y(length), pos_z(length) { }

    // constant across the group variables ---
    size_t length;
    // type
    // source_id

    // individual particle variables ---------
    Vector pos_x;
    Vector pos_y;
    Vector pos_z;
    //Vector vel_u;
    //Vector vel_v;
    //Vector vel_w;
    Vector birthtime;       // TODO: specialize this to be a Vector<time_t>
    // has_deposited        // TODO: specialize this to be a Vector<bool>
    // ...

    // convenience functions -----------------
    template <typename TOther>
    Particles<Vector>& operator=(const TOther &other) {
        pos_x = other.pos_x;
        pos_y = other.pos_y;
        pos_z = other.pos_z;

        return *this;
    }
};


/// ------ Debugging utils ------------------
template <typename Particles>
void ParticlesPrint(Particles &p, std::ostream &out) {
    for (size_t i=0; i<p.length; i++) {
        out << p.pos_x[i] << " "
            << p.pos_y[i] << " "
            << p.pos_z[i] << std::endl;
    }
}


/// ------ Randomize Positions ---------------
template <typename Particles>
struct ParticlesRandomizePositionFunctor {
    ParticlesRandomizePositionFunctor(float xmin_, float xmax_, float ymin_, float ymax_, float zmin_, float zmax_) :
        xmin(xmin_), xmax(xmax_),
        ymin(ymin_), ymax(ymax_),
        zmin(zmin_), zmax(zmax_) { }

    const float xmin, xmax;
    const float ymin, ymax;
    const float zmin, zmax;

    __host__ __device__
    void operator()(typename Particles::ref p) const {
        p.pos_x = xmax * (float)drand48();
        p.pos_y = ymax * (float)drand48();
        p.pos_z = zmax * (float)drand48();
    }
};


template <typename Particles>
void ParticlesRandomizePosition(Particles &p,
        typename Particles::T xmin, typename Particles::T xmax,
        typename Particles::T ymin, typename Particles::T ymax,
        typename Particles::T zmin, typename Particles::T zmax)
{
    thrust::for_each(
            p.begin(),
            p.end(),
            ParticlesRandomizePositionFunctor<Particles>(xmin, xmax, ymin, ymax, zmin, zmax)
    );
}



/// ------ Fill positions with a constant ----
template <typename Particles>
void ParticlesFillPosition(Particles &p,
        typename Particles::T x,
        typename Particles::T y,
        typename Particles::T z)
{
    // TODO: this may be able to be made faster with a single functor
    thrust::fill(p.pos_x.begin(), p.pos_x.end(), x);
    thrust::fill(p.pos_y.begin(), p.pos_y.end(), y);
    thrust::fill(p.pos_z.begin(), p.pos_z.end(), z);
}


/// ------ Fill births based on start/stop/rate vars
template <typename Particles>
void ParticlesFillBirthTime(Particles &p, time_t start, time_t stop, float rate)
{
    // rate = p / s
    float step = rate * (stop - start);
    thrust::sequence(p.birthtime.begin(), p.birthtime.end(), (float)start, step);
}



#endif /* end of include guard: PARTICLES_CUH */
