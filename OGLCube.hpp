#ifndef OGLCUBE_HPP
#define OGLCUBE_HPP

#include <iostream>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "OGLShaderManager.hpp"

class OGLCube {
    public:
        OGLCube(const float origin_x, const float origin_y, const float origin_z,
                const float size_x,   const float size_y,   const float size_z);

        ~OGLCube() {
            glDeleteBuffers(1, &vertex_buffer);
        }

        void draw(const glm::mat4 &mvpMat);

    private:
        GLuint vertex_buffer;
        static GLuint index_buffer;
        static const unsigned int indices[24];
        static GLuint programID;
        static GLuint shaderMVP_loc;

        static void static_init();
};

#endif /* end of include guard: OGLCUBE_HPP */

