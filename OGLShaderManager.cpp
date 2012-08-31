#include "OGLShaderManager.hpp"

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>


// initialize the static shared manager
OGLShaderManager OGLShaderManager::SHARED_MGR;


namespace {
    std::string read_file(const char *path) {
        std::string contents;

        std::ifstream fstream(path, std::ios::in);
        if (fstream.is_open()) {
            std::string line = "";
            while (getline(fstream, line)) {
                contents += "\n" + line;
            }
            fstream.close();
        }

        return contents;
    }

    GLint compile_shader(GLuint &shader_id, const char *code) {
        GLint success = GL_FALSE;
        glShaderSource(shader_id, 1, &code, NULL);
        glCompileShader(shader_id);
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
        return success;
    }

    GLint link_program(GLuint &prog_id, GLuint &vert_id, GLuint &frag_id) {
        GLint success = GL_FALSE;
        glAttachShader(prog_id, vert_id);
        glAttachShader(prog_id, frag_id);
        glLinkProgram(prog_id);
        glGetProgramiv(prog_id, GL_LINK_STATUS, &success);
        return success;
    }

    std::string get_shader_info_log(const GLuint &shader_id) {
        int log_length;
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log(std::max(log_length, 1));
        glGetShaderInfoLog(shader_id, log_length, NULL, &log[0]);

        return std::string(log.begin(), log.end());
    }

    std::string get_program_info_log(const GLuint &prog_id) {
        int log_length;
        glGetProgramiv(prog_id, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log(std::max(log_length, 1));
        glGetProgramInfoLog(prog_id, log_length, NULL, &log[0]);

        return std::string(log.begin(), log.end());
    }
};


GLuint compile_shader_program(const std::string &vert_path, const std::string &frag_path)
{
    // create the shader refs
    GLuint vert_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag_id = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint prog_id = glCreateProgram();

    // TODO: need to check for creation errors here?

    std::string vert_code = read_file(vert_path.c_str());
    std::string frag_code = read_file(frag_path.c_str());

    GLint success = GL_FALSE;

    // compile the vertex shader
    success = compile_shader(vert_id, vert_code.c_str());
    if (success != GL_TRUE) {
        std::cerr << "error compiling vertex shader (" << vert_path << "): " << std::endl
                  << get_shader_info_log(vert_id) << std::endl;
        goto shader_cleanup;   // goto cleanup pattern - not necessarily bad!
    }

    // compile the fragment shader
    success = compile_shader(frag_id, frag_code.c_str());
    if (success != GL_TRUE) {
        std::cerr << "error compiling fragment shader (" << frag_path << "): " << std::endl
                  << get_shader_info_log(frag_id) << std::endl;
        goto shader_cleanup;   // goto cleanup pattern - not necessarily bad!
    }

    // link the shader program
    success = link_program(prog_id, vert_id, frag_id);
    if (success != GL_TRUE) {
        std::cerr << "error linking shader program: " << get_program_info_log(prog_id) << std::endl;
        glDeleteProgram(prog_id);
        prog_id = 0;            // set to OpenGL invalid value
        goto shader_cleanup;    // goto cleanup pattern - not necessarily bad!
    }

shader_cleanup:
    glDeleteShader(vert_id);
    glDeleteShader(frag_id);

    return prog_id;
}



namespace {
    std::vector<std::string> get_file_names(const char *dirpath)
    {
        std::vector<std::string> files;
        DIR* dir = opendir(dirpath);
        dirent *dirp;
        if (dir == NULL) {
            std::cerr << "error opening " << dirpath <<": " << strerror(errno) << std::endl;
            goto cleanup;
        }

        while ((dirp = readdir(dir)) != NULL) {
            std::string fname(dirp->d_name);
            if (fname.compare(".") == 0)
                continue;
            if (fname.compare("..") == 0)
                continue;
            files.push_back(fname);
        }

    cleanup:
        closedir(dir);
        return files;
    }
};

void OGLShaderManager::add_directory(const char * dirpath)
{
    typedef std::vector<std::string> FileList;
    typedef FileList::iterator       Iterator;

    FileList files = get_file_names(dirpath);

    // look for file pairs that contain a .vert.glsl and a .frag.glsl suffix
    //  and add them to the map

    sort(files.begin(), files.end());
    // file list now sorted so .frag.glsl will be immediately before .vert.glsl
    for (Iterator it=files.begin(); it<files.end()-1; ++it) {
        const std::string &f1 = *it;
        const std::string &f2 = *(it+1);

        const size_t f1_l = f1.length();
        const size_t f2_l = f2.length();

        if (f1_l < 11 || f1_l != f2_l)
            continue;
        if (f1.compare(f1_l-10, 10, ".frag.glsl") != 0)
            continue;
        if (f2.compare(f2_l-10, 10, ".vert.glsl") != 0)
            continue;

        if (f1.compare(0, f1_l-10, f2, 0, f1_l-10) != 0)
            continue;


        const std::string name = f1.substr(0, f1_l-10);

        std::string frag_path = std::string(dirpath) + "/" + f1;
        std::string vert_path = std::string(dirpath) + "/" + f2;

        GLuint prog_id = compile_shader_program(vert_path, frag_path);
        if (prog_id != 0) {
            program_ids[name] = prog_id;
        }
    }
}


GLuint OGLShaderManager::get_program_id(const std::string &name)
{
    typedef std::map<std::string, GLuint>::iterator Iterator;

    Iterator it = program_ids.find(name);
    if (it == program_ids.end()) {
        return 0;
    }
    return (*it).second;
}


void OGLShaderManager::print_program_ids()
{
    typedef std::map<std::string, GLuint>::iterator Iterator;

    for (Iterator it=program_ids.begin(); it!=program_ids.end(); ++it) {
        std::cout << (*it).first << ": " << (*it).second << std::endl;
    }
}


