#ifndef OGLCONTROLLER_HPP
#define OGLCONTROLLER_HPP


#include <GL/glew.h>
#include <GL/glfw.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OGLController {
    public:
        void look_at(const float pos_x,   const float pos_y,   const float pos_z,
                     const float focus_x, const float focus_y, const float focus_z,
                     const float up_x,    const float up_y,    const float up_z);

        void set_perspective(float fov, float aspect_ratio, float z_near, float z_far);

        void draw();

        const glm::mat4& get_mvp_mat4() {
            return mvpMat;
        }

    private:
        void recalc_mvp() {
            // update model-view-projection matrix
            //glm::mat4 modelMat = glm::mat4(1.0f);
            //mvpMat = projMat * viewMat * modelMat;
            mvpMat = projMat * viewMat;
        }

        //glm::vec3 pos;
        glm::mat4 viewMat;
        glm::mat4 projMat;
        glm::mat4 mvpMat;
};



#endif /* end of include guard: OGLCONTROLLER_HPP */
