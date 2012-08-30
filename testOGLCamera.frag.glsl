#version 330 core

// interpolated values from the vertex shaders
in vec3 fragmentColor;

out vec3 color;

void main() {
    color = fragmentColor;
}
