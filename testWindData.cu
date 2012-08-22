/*
   Author: Danny George
   High Performance Simulation Laboratory
   Boise State University
 
   Permission is hereby granted, free of charge, to any person obtaining a copy of
   this software and associated documentation files (the "Software"), to deal in
   the Software without restriction, including without limitation the rights to
   use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is furnished to do
   so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#include <iostream>

#include <GL/glew.h>
#include <GL/glfw.h>

#include <cuda.h>
#include <cuda_gl_interop.h>


#include <thrust/device_vector.h>

#include "WindData.cuh"


using namespace std;

int main(int argc, char const *argv[])
{
    bool load_file = false;
    const char * file_name;
    if (argc > 1) {
        file_name = argv[1];
        load_file = true;
    }

    // initialize glfw
    if (!glfwInit()) {
        throw std::runtime_error("Couldn't initialize glfw");
    }
    // open a window and create its opengl context
    if (!glfwOpenWindow(16, 16, 0, 0, 0, 0, 32, 0, GLFW_WINDOW)) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
    }
    // initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Couldn't initialize GLEW");
    }

    cudaGLSetGLDevice(0);

    WindDataShape shape(4, 4, 4, 2);

    WindDataThrustHost host(shape);
    if (load_file) {
        std::ifstream in;
        in.open(file_name);
        WindDataThrustASCIIConverter::fill_from_stream(host, in);
        in.close();
    }

    //WindDataThrustDevice  dev(host.shape);
    WindDataTextureMemory tx(host.shape);

    copy(host, tx);

    WindDataThrustHost host2(host.shape);
    copy(tx, host2);


    WindDataThrustASCIIConverter::encode(host, std::cout);
    cout << endl;
    cout << endl;
    cout << endl;
    cout << "-----------------------" << endl;
    cout << endl;
    cout << endl;
    cout << endl;
    WindDataThrustASCIIConverter::encode(host2, std::cout);

    return 0;
}
