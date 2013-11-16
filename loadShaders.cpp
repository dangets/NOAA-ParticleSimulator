#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <cstdio>

#include <GL/glew.h>


#include "loadShaders.hpp"


GLuint loadShaders(const char *vertexFilePath, const char *fragmentFilePath)
{
    // create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // read the vertex shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertexFilePath, std::ios::in);
    if (VertexShaderStream.is_open()) {
        std::string line = "";
        while (getline(VertexShaderStream, line)) {
            VertexShaderCode += "\n" + line;
        }
        VertexShaderStream.close();
    }

    // read the fragment shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragmentFilePath, std::ios::in);
    if (FragmentShaderStream.is_open()) {
        std::string line = "";
        while (getline(FragmentShaderStream, line)) {
            FragmentShaderCode += "\n" + line;
        }
        FragmentShaderStream.close();
    }


    GLint result = GL_FALSE;
    int InfoLogLength;

    // compile vertex shader
    fprintf(stderr, "compiling shader: %s\n", vertexFilePath);
    char const *vertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &vertexSourcePointer, NULL);
    glCompileShader(VertexShaderID);

    // check vertex shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &result);
    if (result != GL_TRUE) {
        glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        std::vector<char> VertexShaderErrorMessage(InfoLogLength);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
    }

    // compile fragment shader
    fprintf(stderr, "compiling shader: %s\n", fragmentFilePath);
    char const *FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
    glCompileShader(FragmentShaderID);

    // check fragment shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &result);
    if (result != GL_TRUE) {
        glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
    }

    // link the program
    fprintf(stderr, "linking program...\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
    if (result != GL_TRUE) {
        glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
        std::vector<char> ProgramErrorMessage(std::max(InfoLogLength, int(1)));
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
    }

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}
