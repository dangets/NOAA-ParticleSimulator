#include <cstdio>
#include <cstdlib> 
#include <ctime>
#include <iostream>

#include <GL/glew.h>
#include <GL/glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Config.hpp"
#include "ConfigJSON.hpp"
#include "loadShaders.hpp"
#include "ParticleSource.hpp"
#include "Particles.cuh"
#include "WindData.cuh"

using std::time_t;


/// ----------------------------------------------
// constants
//static const unsigned int g_window_width = 512;
//static const unsigned int g_window_height = 512;

static glm::vec3 g_pos = glm::vec3(32.0f, 32.0f, 75.0f);
static glm::mat4 g_viewMat;
static glm::mat4 g_projMat = glm::perspective(60.0f, 4.0f/3.0f, 0.1f, 100.0f);


inline glm::mat4 getViewMatrix() {
    return g_viewMat;
}

inline glm::mat4 getProjectionMatrix() {
    return g_projMat;
}

// mouse controls
//static int g_mouse_old_x;
//static int g_mouse_old_y;
//static int g_mouse_buttons = 0;
//static float g_rotate_x = 0.0;
//static float g_rotate_y = 0.0;


time_t g_t = 0;


void initGL(int argc, char** argv)
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



/// ----------------------------------------------

template <typename Particles>
void initializeParticlesFromSource(Particles &p, const ParticleSource &src)
{
    ParticlesFillBirthTime(p, src.release_start, src.release_stop, src.release_rate);

    //ParticlesFillPosition(p, src.pos_x, src.pos_y, src.pos_z);
    ParticlesRandomizePosition(p,
            src.pos_x - src.dx, src.pos_x + src.dx,
            src.pos_y - src.dy, src.pos_y + src.dy,
            src.pos_z - src.dz, src.pos_z + src.dz
    );
}


timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}



int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    std::ifstream cfg_file("config.json");
    Config cfg = ConfigFromJSON(cfg_file);

    //////////////////////////////////////////////////////////

    initGL(argc, argv);
    cudaGLSetGLDevice(0);

    // Create and compile our GLSL program from the shaders
    GLuint programID = loadShaders("simpleVertexShader.vert.glsl", "simpleFragmentShader.frag.glsl");

    // send the mvpMat to GLSL code
    GLuint shaderMVP = glGetUniformLocation(programID, "MVP");

    g_viewMat = glm::lookAt(
            g_pos,
            glm::vec3(32.0f, 32.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
    );
    g_projMat = glm::perspective(60.0f, 4.0f/3.0f, 0.1f, 100.0f);

    //////////////////////////////////////////////////////////

    //cudaEvent_t time1cuda, time2cuda;
    //float elapsedTimeCuda;
    //timespec time1c, time2c, elapsedTimeC;
    //cudaEventCreate(&time1cuda);
    //cudaEventCreate(&time2cuda);

    //int num_timesteps = 1000;

    // read wind data from file
    HostWindData host_wind = WindDataFromASCII(argv[1]);

    // create and copy wind data on device
    DeviceWindData dev_wind(host_wind);

    // create a particle source
    ParticleSource &src = cfg.particle_sources[0];

    // create particles array on host of total particles released by source
    //HostParticles host_particles(src.lifetimeParticlesReleased());
    DeviceParticles dev_particles(src.lifetimeParticlesReleased());
    initializeParticlesFromSource(dev_particles, src);
    //dev_particles = host_particles;

    //std::printf("running on host -----------------------\n");
    //cudaEventRecord(time1cuda, 0);
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1c);
    //for (time_t t=0; t<num_timesteps; t++) {
    //    advectParticles(host_particles, host_wind, t);

    //    //char ofname[256];
    //    //std::snprintf(ofname, 255, "data/device_output_%04d.particles", (int)t);
    //    //std::ofstream out(ofname);
    //    //ParticlesPrintActive(dev_particles, out, t);
    //    //out.close();
    //}
    //cudaEventRecord(time2cuda, 0);
    //cudaEventSynchronize(time2cuda);
    //cudaEventElapsedTime(&elapsedTimeCuda, time1cuda, time2cuda);
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2c);
    //elapsedTimeC = diff(time1c, time2c);
    //std::printf("elapsedTimeC:    %ld:%ld\n", elapsedTimeC.tv_sec, elapsedTimeC.tv_nsec);
    //std::printf("elapsedTimeCuda: %g\n", elapsedTimeCuda);

    //std::printf("running on device ---------------------\n");
    //cudaEventRecord(time1cuda, 0);
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1c);
    //for (time_t t=0; t<num_timesteps; t++) {
    //    advectParticles(dev_particles, dev_wind, t);

    //    //char ofname[256];
    //    //std::snprintf(ofname, 255, "data/device_output_%04d.particles", (int)t);
    //    //std::ofstream out(ofname);
    //    //ParticlesPrintActive(dev_particles, out, t);
    //    //out.close();
    //}
    //cudaEventRecord(time2cuda, 0);
    //cudaEventSynchronize(time2cuda);
    //cudaEventElapsedTime(&elapsedTimeCuda, time1cuda, time2cuda);
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2c);
    //elapsedTimeC = diff(time1c, time2c);
    //std::printf("elapsedTimeC:    %ld:%ld\n", elapsedTimeC.tv_sec, elapsedTimeC.tv_nsec);
    //std::printf("elapsedTimeCuda: %g\n", elapsedTimeCuda);

    //cudaEventDestroy(time1cuda);
    //cudaEventDestroy(time2cuda);

    OpenGLParticles ogl_particles(dev_particles.length);

    // ensure we can capture the escape key being pressed below
    glfwEnable(GLFW_STICKY_KEYS);
    glfwEnable(GLFW_STICKY_MOUSE_BUTTONS);

    // start rendering mainloop
    while (true) {
        bool breakloop = false;
        // loop break events
        if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
            breakloop = true;
        if (!glfwGetWindowParam(GLFW_OPENED))
            breakloop = true;

        // pause on spacebar (somewhat hackish)
        if (glfwGetKey(GLFW_KEY_SPACE) == GLFW_PRESS) {
            glfwSleep(0.1);
            while (glfwGetKey(GLFW_KEY_SPACE) != GLFW_RELEASE) {
                glfwPollEvents();
            }

            // key is released, wait for it to be pressed again
            while (glfwGetKey(GLFW_KEY_SPACE) == GLFW_RELEASE) {
                if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS) {
                    breakloop = true;
                    break;
                }

                glfwWaitEvents();
            }

            // wait for it to be released again
            glfwSleep(0.1);
            while (glfwGetKey(GLFW_KEY_SPACE) != GLFW_RELEASE) {
                glfwPollEvents();
            }
        }

        if (breakloop)
            break;

        // step the simulation
        advectParticles(dev_particles, dev_wind, g_t);
        ogl_particles.copy(dev_particles);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glEnable(GL_DEPTH_TEST);  // TODO: measure performance of depth test enabled...

        glEnable(GL_POINT_SPRITE);
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        // use setup shaders
        glUseProgram(programID);

        // update model-view-projection matrix
        //computeMatricesFromInputs();
        glm::mat4 projMat = getProjectionMatrix();
        glm::mat4 viewMat = getViewMatrix();
        glm::mat4 modelMat = glm::mat4(1.0f);
        glm::mat4 mvpMat = projMat * viewMat * modelMat;

        // upload transformation to currently bound shader
        glUniformMatrix4fv(shaderMVP, 1, GL_FALSE, &mvpMat[0][0]);

        // render vertices (attribute 0 specified in shader)
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, ogl_particles.pos_vbo);

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
        glDrawArrays(GL_POINTS, 0, ogl_particles.length);
        glDisableVertexAttribArray(0);

        glUseProgram(0);
        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDisable(GL_POINT_SPRITE);
        //glDisable(GL_DEPTH_TEST);

        // swap buffers
        glfwSwapBuffers();
        g_t += 1;
    }

	glDeleteProgram(programID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}
