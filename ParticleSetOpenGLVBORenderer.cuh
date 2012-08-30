#ifndef PARTICLESETOPENGLVBORENDERER_CUH
#define PARTICLESETOPENGLVBORENDERER_CUH


#include <GL/glew.h>
#include <glm/glm.hpp>

#include "loadShaders.hpp"
#include "ParticleSetOpenGLVBO.cuh"

class ParticleSetOpenGLVBORenderer {
    public:
        void init() {
            if (!did_static_init) {
                // Create and compile our GLSL program from the shaders
                programID = loadShaders(
                        "ParticleSetOpenGLVBORenderer.vert.glsl",
                        "ParticleSetOpenGLVBORenderer.frag.glsl"
                    );
                shaderMVP_loc = glGetUniformLocation(programID, "MVP");
                did_static_init = true;
            }
        }


        void draw(const glm::mat4 &mvpMat, const ParticleSetOpenGLVBO &particles);

    private:
        static bool did_static_init;
        static GLuint programID;
        static GLuint shaderMVP_loc;
};



#endif /* end of include guard: PARTICLESETOPENGLVBORENDERER_CUH */
