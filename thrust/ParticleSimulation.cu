#include <cstdio>
#include <cstdlib> 
#include <ctime>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ParticleSource.hpp"
#include "Particles.cuh"
#include "WindData.cuh"

using std::time_t;


/// ----------------------------------------------
// constants
const unsigned int g_window_width = 512;
const unsigned int g_window_height = 512;

// mouse controls
int g_mouse_old_x, g_mouse_old_y;
int g_mouse_buttons = 0;
float g_rotate_x = 0.0;
float g_rotate_y = 0.0;
float g_translate_z = -3.0;

DeviceParticles *g_dev_particles;
DeviceWindData *g_dev_wind;
OpenGLParticles *g_ogl_particles;
time_t g_t = 0;


void display(void)
{
    if (g_dev_particles == NULL) {
        return;
    }

    advectParticles(*g_dev_particles, *g_dev_wind, g_t);

    (*g_ogl_particles).copy(*g_dev_particles);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_translate_z);
    glRotatef(g_rotate_x, 1.0, 0.0, 0.0);
    glRotatef(g_rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, (*g_ogl_particles).pos_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, (*g_ogl_particles).length);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();

    g_t += 1;
}


void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        g_mouse_buttons |= 1<<button;
    } else if(state == GLUT_UP) {
        g_mouse_buttons = 0;
    }

    g_mouse_old_x = x;
    g_mouse_old_y = y;

    glutPostRedisplay();
}


void motion(int x, int y)
{
    float dx, dy;
    dx = x - g_mouse_old_x;
    dy = y - g_mouse_old_y;

    if(g_mouse_buttons & 1) {
        g_rotate_x += dy * 0.2;
        g_rotate_y += dx * 0.2;
    } else if (g_mouse_buttons & 4) {
        g_translate_z += dy * 0.01;
    }

    g_mouse_old_x = x;
    g_mouse_old_y = y;
}


void keyboard(unsigned char key, int, int)
{
    switch(key) {
        case (27):
            exit(0);
        default:
            break;
    }
}


void initGL(int argc, char** argv)
{
    // Create GL context
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(g_window_width, g_window_height);
    glutCreateWindow("Thrust/GL interop");

    GLenum glewInitResult = glewInit();
    if (glewInitResult != GLEW_OK) {
        throw std::runtime_error("Couldn't initialize GLEW");
    }

    // initialize GL
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, g_window_width, g_window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)g_window_width / (GLfloat)g_window_height, 0.1, 100.0);
    gluLookAt(32.0f, 32.0f, 75.0f,
              32.0f, 32.0f, 0.0f,
              0.0f, 1.0f, 0.0f);

    
}




/// ----------------------------------------------

HostParticles createParticlesFromSource(const ParticleSource &src)
{
    // create particles of number of lifetime particles
    HostParticles host_particles(src.lifetimeParticlesReleased());

    // initialize position and birth times from source info
    // set position to randomly within the source cube
    ParticlesRandomizePosition(host_particles,
            src.pos_x - src.dx, src.pos_x + src.dx,
            src.pos_y - src.dy, src.pos_y + src.dy,
            src.pos_z - src.dz, src.pos_z + src.dz
    );

    // initialize the birth times based on rate of release
    ParticlesFillBirthTime(host_particles, src.release_start, src.release_stop, src.release_rate);

    return host_particles;
}


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

    //////////////////////////////////////////////////////////

    initGL(argc, argv);
    cudaGLSetGLDevice(0);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    //////////////////////////////////////////////////////////

    cudaEvent_t time1cuda, time2cuda;
    float elapsedTimeCuda;

    timespec time1c, time2c, elapsedTimeC;

    int num_timesteps = 1000;

    cudaEventCreate(&time1cuda);
    cudaEventCreate(&time2cuda);

    // read wind data from file
    HostWindData host_wind = WindDataFromASCII(argv[1]);

    // create and copy wind data on device
    DeviceWindData dev_wind(host_wind);

    // create a particle source
    ParticleSource src(
            20,  20,   13,  // position
             0, 800, 2000,  // start, stop, rate
             1,   1,    1   // dx, dy, dz
        );

    // create particles array on host of total particles released by source
    //HostParticles host_particles(src.lifetimeParticlesReleased());
    DeviceParticles dev_particles(src.lifetimeParticlesReleased());
    initializeParticlesFromSource(dev_particles, src);
    //dev_particles = host_particles;

    g_dev_particles = &dev_particles;
    g_dev_wind = &dev_wind;

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

    g_ogl_particles = new OpenGLParticles(g_dev_particles->length);
    // start rendering mainloop
    glutMainLoop();

    return 0;
}
