//
// Created by cytech on 24/09/24.
//

#ifndef GL2_SHADER_CLASS_H
#define GL2_SHADER_CLASS_H

#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <string>

std::string get_file_contents(const char *filename);

class shader_class {
public:
    GLuint ID;

    shader_class(const char *vertexFile, const char *fragmentFile);

    void Activate();

    void Delete();

private:
    void compileErrors(unsigned int shader, const char *type);
};


#endif //GL2_SHADER_CLASS_H