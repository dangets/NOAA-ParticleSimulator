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

#include "ParticleSetOpenGLVBO.cuh"


// Make a shallow copy of 'from' into 'to'
void copy(const ParticleSetOpenGLVBO &from, ParticleSetOpenGLVBO &to) {
    // self-copy check
    if (&from == &to) { return; }
    to.m_size = from.m_size;
    to.pos_vbo = from.pos_vbo;
}



/**
 * functor object to copy from 3 separate vectors into one interleaved chunk
 */
struct ParticlePosToFloat3 {
    template <typename Tuple>
    __host__ __device__
    float3 operator()(Tuple tup) const {
        return make_float3(
                thrust::get<0>(tup),    // pos_x
                thrust::get<1>(tup),    // pos_y
                thrust::get<2>(tup)     // pos_z
        );
    }
};


/**
 * NOTE: copying from ParticleSetThrustHost is *possible*, but not
 *  allowed for the sake that efficiency isn't accidentally overlooked
 */
void copy(const ParticleSetThrustDevice &from_thrust, ParticleSetOpenGLVBO &to_vbo) {
    if (from_thrust.size() != to_vbo.size()) {
        throw std::invalid_argument("ParticleSet 'from' and 'to' sizes do not match!");
    }

    float3 *raw_ptr;
    std::size_t buf_size;

    to_vbo.mapCUDA();

    cudaGraphicsResourceGetMappedPointer((void **)&raw_ptr, &buf_size, to_vbo.position_cuda_vbo());
    thrust::device_ptr<float3> dev_ptr = thrust::device_pointer_cast(raw_ptr);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(from_thrust.pos_x.begin(), from_thrust.pos_y.begin(), from_thrust.pos_z.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(from_thrust.pos_x.end(),   from_thrust.pos_y.end(),   from_thrust.pos_z.end())),
        dev_ptr,
        ParticlePosToFloat3()
    );

    to_vbo.unmapCUDA();
}


