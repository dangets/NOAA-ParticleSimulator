#include "OGLCube.hpp"

// setup the static variables
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
GLuint OGLCube::programID = 0;
GLuint OGLCube::shaderMVP_loc = 0;


void OGLCube::static_init()
{
    static bool did_init = false;
    if (did_init) {
        return;
    }

    // get the id of the previously compiled shaders
    programID = OGLShaderManager::SHARED_MGR.get_program_id("flat");
    shaderMVP_loc = glGetUniformLocation(programID, "MVP");

    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), &indices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    did_init = true;
}



OGLCube::OGLCube(const float origin_x, const float origin_y, const float origin_z,
                 const float size_x,   const float size_y,   const float size_z)
{
    static_init();

    const float &ox = origin_x;
    const float &oy = origin_y;
    const float &oz = origin_z;

    const float &sx = size_x;
    const float &sy = size_y;
    const float &sz = size_z;

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



void OGLCube::draw(const glm::mat4 &mvpMat)
{
    glUseProgram(programID);

    // upload transformation to currently bound shader
    glUniformMatrix4fv(shaderMVP_loc, 1, GL_FALSE, &mvpMat[0][0]);

    // render vertices (attribute 0 specified in shader)
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

    // 1st attribute buffer: vertices
    glVertexAttribPointer(
            0,          // attribute 0 (no particular reason)
            3,          // size
            GL_FLOAT,   // type
            GL_FALSE,   // normalized?
            0,          // stride
            (void *)0   // array buffer offset
    );

    //glEnableClientState(GL_VERTEX_ARRAY);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, (GLvoid *)NULL);

    // draw the shape
    //glDrawArrays(GL_LINES, 0, 12);
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
