//
// Created by zelie on 10/7/24.
//

#ifndef TEXTURE_H
#define TEXTURE_H
#include <GL/glew.h>

#include "shader_class.h"

namespace gl3 {

class texture {
    public:
    GLuint id;

    texture(const char* image, int width, int height);
    texture(int width, int height);
    texture();

    static void tex_unit(shader_class& shader, const char* uniform, GLuint unit);
    void bind() const;
    static void unbind();

    ~texture();
};

} // gl3

#endif //TEXTURE_H
