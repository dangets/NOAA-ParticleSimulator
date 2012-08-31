
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>
#include <libgen.h>     // basename

#include <GL/glew.h>
#include <GL/glfw.h>
#include <cuda_gl_interop.h>

#include "loadShaders.hpp"
#include "CUDATimer.cuh"

#include "OGLController.hpp"
#include "OGLCube.hpp"
#include "OGLShaderManager.hpp"
#include "Config.hpp"
#include "ConfigJSON.hpp"
#include "WindData.cuh"
#include "ParticleSet.cuh"
#include "ParticleSetOpenGLVBO.cuh"
#include "ParticleSetOpenGLVBORenderer.cuh"
#include "advect_original.cuh"
#include "advect_runge_kutta.cuh"

#include "vtk_io.cuh"


namespace {
    // ugly globals that should be changeable in the future
    const int   SCREEN_WIDTH  = 1024;
    const int   SCREEN_HEIGHT = 768;
    const float SCREEN_ASPECT = float(SCREEN_WIDTH) / SCREEN_HEIGHT;

    // get the directory of the EXECUTABLE (this is different than cwd)
    std::string get_exec_dir() {
        size_t buf_size = 256;
        std::vector<char> buf(buf_size);
        size_t read_size = readlink("/proc/self/exe", &(buf[0]), buf_size-1);
        // TODO: check read_size <= buf_size-1 for possible truncation
        buf[read_size] = '\0';

        return std::string(dirname(&(buf[0]))) + "/";
    }


    void init_OpenGL(const std::string &shader_dirpath="shaders/")
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
        if (!glfwOpenWindow(SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, 0, 0, 32, 0, GLFW_WINDOW)) {
            fprintf(stderr, "Failed to open GLFW window\n");
            glfwTerminate();
        }

        // initialize GLEW
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("Couldn't initialize GLEW");
        }

        glfwSetWindowTitle("NOAA Particle Simulation");

        // initialize GL
        glClearColor(0.0f, 0.0f, 0.1f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        // initialize the shared shader manager
        OGLShaderManager::SHARED_MGR.add_directory(shader_dirpath.c_str());
    }

    void cleanup_OpenGL()
    {
        glfwTerminate();
    }


    void init_CUDA()
    {
        cudaGLSetGLDevice(0);
    }
};




int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::printf("Usage: %s dataFile\n", argv[0]);
        std::exit(1);
    }

    const std::string exe_dir = get_exec_dir();
    std::string cfg_file   = exe_dir + "config.json";
    std::string shader_dir = exe_dir + "shaders/";

    // NOTE: these init calls MUST happen before any other CUDA function calls
    //  and init_OpenGL MUST be before init_CUDA
    init_OpenGL(shader_dir);
    init_CUDA();

    CUDATimer timer_upload;
    CUDATimer timer_compute;

    // read in the config file
    std::ifstream cfg_istream(cfg_file.c_str());
    Config cfg = ConfigFromJSON(cfg_istream);

    // create a particle source
    ParticleSource &src = cfg.particle_sources[0];

    // read wind data from file
    WindDataThrustHost wind_h = WindDataThrustASCIIConverter::from_file(argv[1]);

    float elapsed;
    int num_iterations = 5000;
    std::cout << "num_iterations: " << num_iterations << std::endl;

    //// --------------------------------------------
    ////  host side ---------------------------------
    //// --------------------------------------------
    //ParticleSetThrustHost part_h = ParticleSetThrustHost_from_particle_source(src);
    //std::cout << "num_particles: " << part_h.size() << std::endl;

    //timer_compute.start();
    //for (int i=0; i<num_iterations; ++i) {
    //    //advect_original(part_h, wind_h, (float)i);
    //    advect_runge_kutta(part_h, wind_h, (float)i);
    //
    //    //if (i % 10 == 0) {
    //    //    std::stringstream fname;
    //    //    fname << "junk." << i << ".vtp";
    //    //    write_vtp(fname.str(), part_h);
    //    //}
    //}
    //elapsed = timer_compute.get_elapsed_time_sync();
    //std::cout << "compute_time: " << elapsed << std::endl;

    // --------------------------------------------
    //  device side -------------------------------
    // --------------------------------------------
    float cam_x = wind_h.shape.x() / 2.0f;
    float cam_y = wind_h.shape.y() / 2.0f;
    float cam_z = std::max(wind_h.shape.x(), wind_h.shape.y()) * 1.2f;

    float foc_x = cam_x;
    float foc_y = cam_y;
    float foc_z = wind_h.shape.z() / 2.0f;

    OGLController ogl_ctrl;
    ogl_ctrl.look_at(cam_x, cam_y, cam_z,  // camera position
                     foc_x, foc_y, foc_z,   // focus point
                     0.0f, 1.0f, 0.0f);     // up-vector
    ogl_ctrl.set_perspective(60.0f, SCREEN_ASPECT, 0.1f, 500.0f);

    OGLCube ogl_border = OGLCube(
            0.0f, 0.0f, 0.0f,
            float(wind_h.shape.x()),
            float(wind_h.shape.y()),
            float(wind_h.shape.z()));

    ParticleSetOpenGLVBORenderer pset_renderer;

    // copy wind data to device
    timer_upload.start();
    WindDataThrustDevice  wind_d(wind_h);
    elapsed = timer_upload.get_elapsed_time_sync();
    std::cout << "wind_upload_time: " << elapsed << std::endl;

    // wind data in texture memory
    timer_upload.start();
    WindDataTextureMemory wind_t(wind_h.shape);
    copy(wind_h, wind_t);
    elapsed = timer_upload.get_elapsed_time_sync();
    std::cout << "wind_texture_upload_time: " << elapsed << std::endl;

    ParticleSetThrustDevice part_d = ParticleSetThrustDevice_from_particle_source(src);
    std::cout << "num_particles: " << part_d.size() << std::endl;

    ParticleSetOpenGLVBO part_ogl(part_d.size());

    // ensure we can capture the escape key being pressed below
    glfwEnable(GLFW_STICKY_KEYS);
    glfwEnable(GLFW_STICKY_MOUSE_BUTTONS);

    timer_compute.start();
    for (int i=0; i<num_iterations; ++i) {
        // loop break events
        if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
            break;
        if (!glfwGetWindowParam(GLFW_OPENED))
            break;

        //advect_original(part_d, wind_d, (float)i);
        //advect_runge_kutta(part_d, wind_d, (float)i);

        //advect_original(part_d, wind_t, (float)i);
        advect_runge_kutta(part_d, wind_t, (float)i);

        if (i % 10 == 0) {
            copy(part_d, part_ogl);
            pset_renderer.draw(ogl_ctrl.get_mvp_mat4(), part_ogl);
            ogl_border.draw(ogl_ctrl.get_mvp_mat4());
            ogl_ctrl.draw();
        }
    }
    elapsed = timer_compute.get_elapsed_time_sync();
    std::cout << "compute_time: " << elapsed << std::endl;

    cleanup_OpenGL();

    return 0;
}
