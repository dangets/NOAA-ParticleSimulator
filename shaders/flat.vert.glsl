#version 330 compatibility

uniform mat4 MVP;
//uniform float pointRadius;
//uniform float pointScale;

layout(location=0) in vec3 vertexPosition_modelspace;
//layout(location=1) in vec3 vertexColor;

//out vec3 fragmentColor;

void main() {
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1);

    //fragmentColor = vertexColor;
    //fragmentColor = vec3(1.0f, 1.0f, 1.0f);
}
