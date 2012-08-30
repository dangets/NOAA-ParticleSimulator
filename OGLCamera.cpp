#include "OGLCamera.hpp"

#include <iostream>

namespace {
    const float PI = acos(-1.0);
    const float DEG2RAD = PI / 180.0f;

    void print_vec3(const glm::vec3 &vec, std::ostream &out=std::cout) {
        out << vec[0] << " " << vec[1] << " " << vec[2] << std::endl;
    }
};


void OGLCamera::rotate_up(float degrees) {
    // rotating about x axis
    float rad = degrees * DEG2RAD;
    float cos_v = cos(rad);
    float sin_v = sin(rad);

    glm::mat4 rot(1,     0,      0, 0,
                    0, cos_v, -sin_v, 0,
                    0, sin_v,  cos_v, 0,
                    0,     0,      0, 1);

    mat = rot * mat;
}


void OGLCamera::rotate_right(float degrees) {
    // rotating about y axis
    float rad = degrees * DEG2RAD;
    float cos_v = cos(rad);
    float sin_v = sin(rad);

    glm::mat4 rot( cos_v, 0, sin_v, 0,
                       0, 1,     0, 0,
                  -sin_v, 0, cos_v, 0,
                       0, 0,     0, 1);

    mat = rot * mat;
}


void OGLCamera::move_right(float amount) {
    glm::mat4 trans(1);
    trans[3][0] = -amount;
    mat = trans * mat;
}


void OGLCamera::move_up(float amount) {
    glm::mat4 trans(1);
    trans[3][1] = -amount;
    mat = trans * mat;
}


void OGLCamera::move_forward(float amount) {
    glm::mat4 trans(1);
    trans[3][2] = amount;
    mat = trans * mat;
}


void OGLCamera::look_at(glm::vec3 pos, glm::vec3 target, glm::vec3 up) {
    mat = glm::lookAt(pos, target, up);
}

