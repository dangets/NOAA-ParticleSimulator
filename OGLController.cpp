#include "OGLController.hpp"

void OGLController::look_at(
        const float pos_x,   const float pos_y,   const float pos_z,
        const float focus_x, const float focus_y, const float focus_z,
        const float up_x,    const float up_y,    const float up_z)
{
    viewMat = glm::lookAt(
            glm::vec3(pos_x, pos_y, pos_z),
            glm::vec3(focus_x, focus_y, focus_z),
            glm::vec3(up_x, up_y, up_z));
    recalc_mvp();
}

void OGLController::set_perspective(float fov, float aspect_ratio, float z_near, float z_far)
{
    projMat = glm::perspective(fov, aspect_ratio, z_near, z_far);
    recalc_mvp();
}

void OGLController::draw() {
    // swap buffers
    glfwSwapBuffers();

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
