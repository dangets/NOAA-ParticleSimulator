
#include <iostream>
#include <sstream>

#include <GL/glew.h>
#include <GL/glfw.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "loadShaders.hpp"

#include "Config.hpp"
#include "ConfigJSON.hpp"
#include "WindData.cuh"
#include "ParticleSet.cuh"
#include "ParticleSetOpenGLVBO.cuh"
#include "advect_original.cuh"
#include "advect_runge_kutta.cuh"

#include "vtk_io.cuh"


void init_OpenGL()
{
    // initialize glfw
    if (!glfwInit()) {
        throw std::runtime_error("Couldn't initialize glfw");
    }

    glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);   // 4x antialiasing
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // open a window and create its opengl context
    if (!glfwOpenWindow(1024, 768, 0, 0, 0, 0, 32, 0, GLFW_WINDOW)) {
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

void cleanup_OpenGL()
{
    glfwTerminate();
}


void init_CUDA()
{
    cudaGLSetGLDevice(0);
}



class ParticleSetOpenGLVBORenderer {
    public:
        GLuint programID;
        GLuint shaderMVP;

        glm::vec3 pos;
        glm::mat4 viewMat;
        glm::mat4 projMat;

        ParticleSetOpenGLVBORenderer()
            : pos(glm::vec3(32.0f, 32.0f, 200.0f)),
              viewMat(glm::lookAt(pos, glm::vec3(32.0f, 32.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f))),
              projMat(glm::perspective(60.0f, 4.0f/3.0f, 0.1f, 1000.0f))
        {
            // Create and compile our GLSL program from the shaders
            programID = loadShaders("simpleVertexShader.vert.glsl", "simpleFragmentShader.frag.glsl");
            shaderMVP = glGetUniformLocation(programID, "MVP");
        }


        void draw(const ParticleSetOpenGLVBO &particles) {
            // Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            //glEnable(GL_DEPTH_TEST);  // TODO: measure performance vs depth test enabled...

            glEnable(GL_POINT_SPRITE);
            glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

            // tell OpenGL to use the shader program
            glUseProgram(programID);

            // update model-view-projection matrix
            //glm::mat4 projMat = getProjectionMatrix();
            //glm::mat4 viewMat = getViewMatrix();
            glm::mat4 modelMat = glm::mat4(1.0f);
            glm::mat4 mvpMat = projMat * viewMat * modelMat;

            // upload transformation to currently bound shader
            glUniformMatrix4fv(shaderMVP, 1, GL_FALSE, &mvpMat[0][0]);

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
            //glDisable(GL_DEPTH_TEST);

            // swap buffers
            glfwSwapBuffers();
        }
};





int main(int argc, char *argv[])
{
    cudaEvent_t start_transfer;
    cudaEvent_t stop_transfer;
    float transfer_time;
    cudaEvent_t start_compute;
    cudaEvent_t stop_compute;
    float compute_time;

    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    init_OpenGL();
    init_CUDA();

    cudaEventCreate(&start_transfer);
    cudaEventCreate(&stop_transfer);
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);

    // read in config file
    std::ifstream cfg_file("config.json");
    Config cfg = ConfigFromJSON(cfg_file);

    // create a particle source
    ParticleSource &src = cfg.particle_sources[0];

    // read wind data from file
    WindDataThrustHost wind_h = WindDataThrustASCIIConverter::from_file(argv[1]);

    int num_iterations = 1000;
    std::cout << "num_iterations: " << num_iterations << std::endl;

    //// --------------------------------------------
    ////  host side ---------------------------------
    //// --------------------------------------------
    //ParticleSetThrustHost part_h = ParticleSetThrustHost_from_particle_source(src);
    //std::cout << "num_particles: " << part_h.size() << std::endl;

    //cudaEventRecord(start_compute, 0);
    //for (int i=0; i<num_iterations; ++i) {
    //    //advect_original(part_h, wind_h, (float)i);
    //    advect_runge_kutta(part_h, wind_h, (float)i);
    //
    //    //if (i % 10 == 0) {
    //    //    std::stringstream fname;
    //    //    fname << "junk." << i << ".vtp";
    //    //    write_vtp(part_h, fname.str());
    //    //}
    //}
    //cudaEventRecord(stop_compute, 0);
    //cudaEventSynchronize(stop_compute);
    //cudaEventElapsedTime(&compute_time, start_compute, stop_compute);
    //std::cout << "compute_time: " << compute_time << std::endl;

    // --------------------------------------------
    //  device side -------------------------------
    // --------------------------------------------
    // copy wind data to device
    cudaEventRecord(start_transfer, 0);
    WindDataThrustDevice  wind_d(wind_h);
    cudaEventRecord(stop_transfer, 0);
    cudaEventSynchronize(stop_transfer);
    cudaEventElapsedTime(&transfer_time, start_transfer, stop_transfer);
    std::cout << "wind_upload_time: " << transfer_time << std::endl;

    // wind data in texture memory
    cudaEventRecord(start_transfer, 0);
    WindDataTextureMemory wind_t(wind_h.shape);
    copy(wind_h, wind_t);
    cudaEventRecord(stop_transfer, 0);
    cudaEventSynchronize(stop_transfer);
    cudaEventElapsedTime(&transfer_time, start_transfer, stop_transfer);
    std::cout << "wind_texture_upload_time: " << transfer_time << std::endl;

    ParticleSetThrustDevice part_d = ParticleSetThrustDevice_from_particle_source(src);
    std::cout << "num_particles: " << part_d.size() << std::endl;

    //ParticleSetOpenGLVBO         part_ogl(part_d.size());
    //ParticleSetOpenGLVBORenderer p_disp;

    // ensure we can capture the escape key being pressed below
    glfwEnable(GLFW_STICKY_KEYS);
    glfwEnable(GLFW_STICKY_MOUSE_BUTTONS);

    cudaEventRecord(start_compute, 0);
    for (int i=0; i<num_iterations; ++i) {
        // loop break events
        if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
            break;
        if (!glfwGetWindowParam(GLFW_OPENED))
            break;

        //advect_original(part_d, wind_d, (float)i);
        advect_runge_kutta(part_d, wind_d, (float)i);

        //advect_original(part_d, wind_t, (float)i);
        //advect_runge_kutta(part_d, wind_t, (float)i);

        //if (i % 10 == 0) {
        //    copy(part_d, part_ogl);
        //    p_disp.draw(part_ogl);
        //}
    }
    cudaEventRecord(stop_compute, 0);
    cudaEventSynchronize(stop_compute);
    cudaEventElapsedTime(&compute_time, start_compute, stop_compute);
    std::cout << "compute_time: " << compute_time << std::endl;


    cleanup_OpenGL();

    return 0;
}
