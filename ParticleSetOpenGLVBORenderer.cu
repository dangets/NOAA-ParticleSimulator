#include "ParticleSetOpenGLVBORenderer.cuh"

// setup the static variables
bool ParticleSetOpenGLVBORenderer::did_static_init = false;
GLuint ParticleSetOpenGLVBORenderer::programID = 0;
GLuint ParticleSetOpenGLVBORenderer::shaderMVP_loc = 0;


void ParticleSetOpenGLVBORenderer::draw(const glm::mat4 &mvpMat, const ParticleSetOpenGLVBO &particles)
{
    //glEnable(GL_DEPTH_TEST);  // TODO: measure performance vs depth test enabled...

    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // tell OpenGL to use the shader program
    glUseProgram(programID);

    // upload transformation to currently bound shader
    glUniformMatrix4fv(shaderMVP_loc, 1, GL_FALSE, &mvpMat[0][0]);

    // render vertices (attribute 0 specified in shader)
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, particles.position_gl_vbo());

    // 1st attribute buffer: vertices
    glVertexAttribPointer(
            0,          // attribute 0 (no particular reason)
            3,          // size
            GL_FLOAT,   // type
            GL_FALSE,   // normalized?
            0,          // stride
            (void *)0   // array buffer offset
    );

    // draw the shape
    glDrawArrays(GL_POINTS, 0, particles.size());
    glDisableVertexAttribArray(0);

    glUseProgram(0);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);
}
