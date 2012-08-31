#ifndef OGLSHADERMANAGER_HPP
#define OGLSHADERMANAGER_HPP

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <GL/glew.h>


class OGLShaderManager {
    public:
        static OGLShaderManager SHARED_MGR;

        void add_directory(const char *dirpath);
        GLuint get_program_id(const std::string &name);

        void print_program_ids();

    private:
        std::map<const std::string, GLuint> program_ids;
};


GLuint compile_shader_program(const std::string &vert_path, const std::string &frag_path);


#endif /* end of include guard: OGLSHADERMANAGER_HPP */

