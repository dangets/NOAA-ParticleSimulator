#ifndef OGLCUBE_HPP
#define OGLCUBE_HPP

#include <iostream>

#include "GL/glew.h"

class OGLCube {
    public:
        OGLCube(const float origin_x, const float origin_y, const float origin_z,
                const float size_x,   const float size_y,   const float size_z)
        {
            static_init();

            const float ox = origin_x;
            const float oy = origin_y;
            const float oz = origin_z;

            const float sx = size_x;
            const float sy = size_y;
            const float sz = size_z;

            GLfloat vertex_buffer_data[] = {
                ox     , oy     , oz     ,
                ox     , oy     , oz + sz,
                ox     , oy + sy, oz     ,
                ox     , oy + sy, oz + sz,
                ox + sx, oy     , oz     ,
                ox + sx, oy     , oz + sz,
                ox + sx, oy + sy, oz     ,
                ox + sx, oy + sy, oz + sz,
            };

            // generate and fill vertex buffer
            glGenBuffers(1, &vertex_buffer);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void draw()
        {
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
            glColor3f(1.0f, 1.0f, 1.0f);
            glDrawElements(
                    GL_LINES,
                    2 * 12,
                    GL_UNSIGNED_INT,
                    (void *)0
            );
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }


    private:
        GLuint vertex_buffer;
        static GLuint index_buffer;
        static const unsigned int indices[24];

        static void static_init();
};

#endif /* end of include guard: OGLCUBE_HPP */

