
#include <cstddef>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using std::size_t;


template <typename T>
class THRUST_HOSTVECTOR_DEFAULT_ALLOCATOR {
    typedef std::allocator<T> Type;
};

template <typename T>
class THRUST_DEVICEVECTOR_DEFAULT_ALLOCATOR {
    typedef thrust::device_malloc_allocator<T> Type;
};



template<template <class, class> class Vector, template <class> class Alloc, typename T>
struct WindDataGeneric {
    const size_t num_x;
    const size_t num_y;
    const size_t num_z;
    const size_t num_t;
    const size_t num_cells;

    Vector<T, Alloc<T> > u;
    Vector<T, Alloc<T> > v;
    Vector<T, Alloc<T> > w;

    WindDataGeneric(size_t x, size_t y, size_t z, size_t t) :
        num_x(x), num_y(y), num_z(z), num_t(t),
        num_cells(x * y * z * t),
        u(num_cells), v(num_cells), w(num_cells)
    { }

    template<typename WindData>
    WindDataGeneric(const WindData &wd) :
        num_x(wd.num_x), num_y(wd.num_y), num_z(wd.num_z), num_t(wd.num_t),
        num_cells(wd.num_cells),
        u(num_cells), v(num_cells), w(num_cells)
    {
        u = wd.u;
        v = wd.v;
        w = wd.w;
    }

    template <typename WindData>
    WindDataGeneric& operator=(const WindData &other) {
        u = other.u;
        v = other.v;
        w = other.w;

        return *this;
    }
};

typedef WindDataGeneric<thrust::host_vector, THRUST_HOSTVECTOR_DEFAULT_ALLOCATOR, float> WindDataHost;
typedef WindDataGeneric<thrust::device_vector, THRUST_DEVICEVECTOR_DEFAULT_ALLOCATOR, float> WindDataDevice;




