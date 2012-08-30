#include "OGLCube.hpp"

const unsigned int OGLCube::indices[] = {
    0, 1,
    2, 3,
    4, 5,
    6, 7,
    0, 2,
    1, 3,
    4, 6,
    5, 7,
    0, 4,
    1, 5,
    2, 6,
    3, 7
};

GLuint OGLCube::index_buffer = 0;


void OGLCube::static_init()
{
    static bool did_init = false;
    if (did_init) {
        return;
    }

    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), &indices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    did_init = true;
}
