#include <iostream>
#include <stdexcept>

#include <GL/glew.h>
#include <GL/glfw.h>

#include "OGLShaderManager.hpp"


void init_OpenGL()
{
    // initialize glfw
    if (!glfwInit()) {
        throw std::runtime_error("Couldn't initialize glfw");
    }

    // open a window and create its opengl context
    if (!glfwOpenWindow(10, 10, 0, 0, 0, 0, 32, 0, GLFW_WINDOW)) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
    }

    // initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Couldn't initialize GLEW");
    }

    glfwSetWindowTitle("Particle Simulation");

    // initialize GL
    glClearColor(0.0f, 0.0f, 0.1f, 0.0f);
    glDisable(GL_DEPTH_TEST);
}


int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shader_dir>" << std::endl;
        exit(1);
    }

    init_OpenGL();

    // using the SHARED_MGR
    OGLShaderManager::SHARED_MGR.add_directory(argv[1]);
    OGLShaderManager::SHARED_MGR.print_program_ids();

    // using an instanced manager
    //OGLShaderManager ogl_smgr;
    //ogl_smgr.add_directory(argv[1]);
    //ogl_smgr.print_program_ids();

    return 0;
}
