#include "OGLCamera.hpp"

#include <GL/glew.h>
#include <GL/glfw.h>
#include "loadShaders.hpp"

// controls code ---------------------
static OGLCamera cam;
static glm::mat4 _projMat;

float speed = 3.0f;
float mouseSpeed = 0.05f;


void computeMatricesFromInputs() {
    // glfwGetTime is called only once (first time this function is called)
    static double lastTime = glfwGetTime();

    // compute time difference between current and last frame
    double curTime = glfwGetTime();
    float deltaTime = float(curTime - lastTime);

    // get and reset mouse pos
    int xpos, ypos;
    glfwGetMousePos(&xpos, &ypos);
    glfwSetMousePos(1024/2, 768/2);

    float mouseDeltaX = float(1024/2 - xpos);
    float mouseDeltaY = float( 768/2 - ypos);

    if (glfwGetMouseButton(GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        // pan position
        cam.move_up(mouseSpeed * mouseDeltaY);
        cam.move_right(-mouseSpeed * mouseDeltaX);
    } else {
        cam.rotate_up(mouseSpeed * mouseDeltaY);
        cam.rotate_right(mouseSpeed * mouseDeltaX);
    }


    // move forward
    if ((glfwGetKey(GLFW_KEY_UP) == GLFW_PRESS) || (glfwGetKey('W') == GLFW_PRESS)) {
        cam.move_forward(deltaTime * speed);
    }
    // move backward
    if ((glfwGetKey(GLFW_KEY_DOWN) == GLFW_PRESS) || (glfwGetKey('S') == GLFW_PRESS)) {
        cam.move_forward(-1 * deltaTime * speed);
    }

    // strafe right
    if ((glfwGetKey(GLFW_KEY_RIGHT) == GLFW_PRESS) || (glfwGetKey('D') == GLFW_PRESS)) {
        cam.move_right(deltaTime * speed);
    }
    // strafe left
    if ((glfwGetKey(GLFW_KEY_LEFT) == GLFW_PRESS) || (glfwGetKey('A') == GLFW_PRESS)) {
        cam.move_right(-1 * deltaTime * speed);
    }

    // strafe up
    if (glfwGetKey('I') == GLFW_PRESS) {
        cam.move_up(deltaTime * speed);
    }
    // strafe down
    if (glfwGetKey('K') == GLFW_PRESS) {
        cam.move_up(-1 * deltaTime * speed);
    }

    lastTime = curTime;
}

// -----------------------------------


int main(int argc, char *argv[])
{
    // initialize glfw
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);   // 4x antialiasing
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // open a window and create its opengl context
    if (!glfwOpenWindow(1024, 768, 0, 0, 0, 0, 32, 0, GLFW_WINDOW)) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    glfwSetWindowTitle("Tutorial 06");
    glClearColor(0.0f, 0.0f, 0.3f, 0.0f);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = loadShaders("testOGLCamera.vert.glsl", "testOGLCamera.frag.glsl");
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");

    cam.look_at(
            glm::vec3(0.0f, 0.0f, 5.0f),    // position
            glm::vec3(0.0f, 0.0f, 0.0f),    // focus
            glm::vec3(0.0f, 1.0f, 0.0f)     // up-vector
    );
    _projMat = glm::perspective(45.0f, 4.0f/3.0f, 0.1f, 100.0f);

    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f,-1.0f,-1.0f,  // triangle 1
        -1.0f,-1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
         1.0f, 1.0f,-1.0f,  // triangle 2
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f,
         1.0f,-1.0f, 1.0f,  // triangle 3
        -1.0f,-1.0f,-1.0f,
         1.0f,-1.0f,-1.0f,
         1.0f, 1.0f,-1.0f,  // triangle 4
         1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f,-1.0f,-1.0f,  // triangle 5
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f,-1.0f,
         1.0f,-1.0f, 1.0f,  // triangle 6
        -1.0f,-1.0f, 1.0f,
        -1.0f,-1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,  // triangle 7
        -1.0f,-1.0f, 1.0f,
         1.0f,-1.0f, 1.0f,
         1.0f, 1.0f, 1.0f,  // triangle 8
         1.0f,-1.0f,-1.0f,
         1.0f, 1.0f,-1.0f,
         1.0f,-1.0f,-1.0f,  // triangle 9
         1.0f, 1.0f, 1.0f,
         1.0f,-1.0f, 1.0f,
         1.0f, 1.0f, 1.0f,  // triangle 10
         1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f,-1.0f,
         1.0f, 1.0f, 1.0f,  // triangle 11
        -1.0f, 1.0f,-1.0f,
        -1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f,  // triangle 12
        -1.0f, 1.0f, 1.0f,
         1.0f,-1.0f, 1.0f
    };

    // One color for each vertex. They were generated randomly.
    static const GLfloat g_color_buffer_data[] = {
        0.583f,  0.771f,  0.014f,
        0.609f,  0.115f,  0.436f,
        0.327f,  0.483f,  0.844f,
        0.822f,  0.569f,  0.201f,
        0.435f,  0.602f,  0.223f,
        0.310f,  0.747f,  0.185f,
        0.597f,  0.770f,  0.761f,
        0.559f,  0.436f,  0.730f,
        0.359f,  0.583f,  0.152f,
        0.483f,  0.596f,  0.789f,
        0.559f,  0.861f,  0.639f,
        0.195f,  0.548f,  0.859f,
        0.014f,  0.184f,  0.576f,
        0.771f,  0.328f,  0.970f,
        0.406f,  0.615f,  0.116f,
        0.676f,  0.977f,  0.133f,
        0.971f,  0.572f,  0.833f,
        0.140f,  0.616f,  0.489f,
        0.997f,  0.513f,  0.064f,
        0.945f,  0.719f,  0.592f,
        0.543f,  0.021f,  0.978f,
        0.279f,  0.317f,  0.505f,
        0.167f,  0.620f,  0.077f,
        0.347f,  0.857f,  0.137f,
        0.055f,  0.953f,  0.042f,
        0.714f,  0.505f,  0.345f,
        0.783f,  0.290f,  0.734f,
        0.722f,  0.645f,  0.174f,
        0.302f,  0.455f,  0.848f,
        0.225f,  0.587f,  0.040f,
        0.517f,  0.713f,  0.338f,
        0.053f,  0.959f,  0.120f,
        0.393f,  0.621f,  0.362f,
        0.673f,  0.211f,  0.457f,
        0.820f,  0.883f,  0.371f,
        0.982f,  0.099f,  0.879f
    };

    // generate and fill vertex buffer
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    // generate and fill color buffer
    GLuint colorBuffer;
    glGenBuffers(1, &colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // ensure we can capture the escape key being pressed below
    glfwEnable(GLFW_STICKY_KEYS);
    glfwEnable(GLFW_STICKY_MOUSE_BUTTONS);

    glfwSetMousePos(1024/2, 768/2);
    do {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use setup shaders
        glUseProgram(programID);

        // update model-view-projection matrix
        computeMatricesFromInputs();

        glm::mat4 projMat = _projMat;
        glm::mat4 viewMat = cam.get_view_mat4();
        glm::mat4 modelMat = glm::mat4(1.0f);
        glm::mat4 mvpMat = projMat * viewMat * modelMat;

        // upload transformation to currently bound shader
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvpMat[0][0]);

        // 1st attribute buffer: vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                0,          // attribute 0 (no particular reason)
                3,          // size
                GL_FLOAT,   // type
                GL_FALSE,   // normalized?
                0,          // stride
                (void *)0   // array buffer offset
        );

        // 2nd attribute buffer: colors
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glVertexAttribPointer(
                1,          // attribute 1 (no particular reason)
                3,          // size
                GL_FLOAT,   // type
                GL_FALSE,   // normalized?
                0,          // stride
                (void *)0   // array buffer offset
        );

        // draw the shape
        glDrawArrays(GL_TRIANGLES, 0, 3*12);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        // swap buffers
        glfwSwapBuffers();
    } while (glfwGetKey(GLFW_KEY_ESC) != GLFW_PRESS &&
            glfwGetWindowParam(GLFW_OPENED));

    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &colorBuffer);
    glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}
