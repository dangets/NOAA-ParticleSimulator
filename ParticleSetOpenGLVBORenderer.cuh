#ifndef PARTICLESETOPENGLVBORENDERER_CUH
#define PARTICLESETOPENGLVBORENDERER_CUH


#include <GL/glew.h>
#include <glm/glm.hpp>

#include "OGLShaderManager.hpp"
#include "ParticleSetOpenGLVBO.cuh"


class ParticleSetOpenGLVBORenderer {
    public:
        ParticleSetOpenGLVBORenderer();
        void draw(const glm::mat4 &mvpMat, const ParticleSetOpenGLVBO &particles);

    private:
        static GLuint programID;
        static GLuint shaderMVP_loc;

        void static_init();
};


#endif /* end of include guard: PARTICLESETOPENGLVBORENDERER_CUH */
