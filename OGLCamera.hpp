#ifndef OGL_CAMERA_HPP
#define OGL_CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


struct OGLCamera {
    void look_at(glm::vec3 pos, glm::vec3 target, glm::vec3 up);

    void rotate_up(float degrees);
    void rotate_right(float degrees);

    void move_right(float amount);
    void move_up(float amount);
    void move_forward(float amount);

    glm::mat4 get_view_mat4() {
        return mat;
    }

    private:
        glm::mat4 mat;
};


#endif /* end of include guard: OGL_CAMERA_HPP */
