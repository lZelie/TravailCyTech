//
// Created by cytech on 24/09/24.
//

#include "vbo.h"

void gl3::vbo::bind() {
    glBindBuffer(GL_ARRAY_BUFFER, ID);
}

void gl3::vbo::unbind() {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

gl3::vbo::~vbo() {
    glDeleteBuffers(1, &ID);
}