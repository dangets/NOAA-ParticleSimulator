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

#ifndef PARTICLESETOPENGLVBO_CUH
#define PARTICLESETOPENGLVBO_CUH

#include <stdexcept>


#include "ParticleSet.cuh"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <boost/tr1/memory.hpp>     // for shared_ptr (standard in C++11)


// only used as an RAII class for ParticleSetOpenGLVBO
//  controls creation and destruction of glBuffer and cudaResource mapping
class ParticlesCUDAVBO {
    public:
        ParticlesCUDAVBO(std::size_t num_bytes)
            : num_bytes(num_bytes)
        {
            // initialize OpenGL buffer and allocate num_bytes
            glGenBuffers(1, &gl_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
            glBufferData(GL_ARRAY_BUFFER, num_bytes, 0, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            GLenum gl_error = glGetError();
            if (gl_error != GL_NO_ERROR) {
                throw std::runtime_error("error initializing OpenGL VBO");
            }

            // register with CUDA
            cudaGraphicsGLRegisterBuffer(&cuda_vbo, gl_vbo, cudaGraphicsMapFlagsWriteDiscard);
        }

        ~ParticlesCUDAVBO() {
            // unregister this buffer object with CUDA
            cudaGraphicsUnregisterResource(cuda_vbo);
            // delete GL buffer
            glDeleteBuffers(1, &gl_vbo);
        }

        // getter functions
        void map()   {   cudaGraphicsMapResources(1, &cuda_vbo, 0); }
        void unmap() { cudaGraphicsUnmapResources(1, &cuda_vbo, 0); }

        // data members --------------
        std::size_t num_bytes;
        GLuint gl_vbo;
        cudaGraphicsResource *cuda_vbo;

        void copy_to(ParticlesCUDAVBO &to) {
            if (to.num_bytes != num_bytes) {
                throw std::invalid_argument("'to' ParticlesCUDAVBO not the same size for copy");
            }

            glBindBuffer(GL_COPY_READ_BUFFER, gl_vbo);
            glBindBuffer(GL_COPY_WRITE_BUFFER, to.gl_vbo);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, num_bytes);
            glBindBuffer(GL_COPY_READ_BUFFER, 0);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        }

    private:
        // disable default copy and equals constructor
        ParticlesCUDAVBO(const ParticlesCUDAVBO &);
        ParticlesCUDAVBO& operator=(const ParticlesCUDAVBO &);
};


class ParticleSetOpenGLVBO {
    public:
        ParticleSetOpenGLVBO(std::size_t size)
            : m_size(size),
              pos_vbo(new ParticlesCUDAVBO(sizeof(float3) * size))
        { }

        // NOTE: the copy constructor creates an object that points to the same pos_vbo (shallow_copy)

        ParticleSetOpenGLVBO deep_copy() const {
            ParticleSetOpenGLVBO to(size());
            pos_vbo->copy_to(*(to.pos_vbo));
            return to;
        }

        std::size_t size() const { return m_size; }

        GLuint                 position_gl_vbo()   const { return pos_vbo->gl_vbo; }
        cudaGraphicsResource * position_cuda_vbo() const { return pos_vbo->cuda_vbo; }

        void mapCUDA()   { pos_vbo->map(); }
        void unmapCUDA() { pos_vbo->unmap(); }

    private:
        std::size_t m_size;
        std::tr1::shared_ptr<ParticlesCUDAVBO> pos_vbo;

        friend void copy(const ParticleSetOpenGLVBO &from, ParticleSetOpenGLVBO &to);
};


void copy(const ParticleSetOpenGLVBO    &from, ParticleSetOpenGLVBO &to);
void copy(const ParticleSetThrustDevice &from, ParticleSetOpenGLVBO &to);




#endif /* end of include guard: PARTICLESETOPENGLVBO_CUH */
