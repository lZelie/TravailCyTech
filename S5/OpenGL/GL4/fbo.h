//
// Created by zelie on 10/10/24.
//

#ifndef FBO_H
#define FBO_H
#include <GL/glew.h>

namespace gl3
{
    class fbo
    {
    public:
        GLuint id{};
        GLuint tex_id{};
        GLuint depth_render_buffer{};
        int width;
        int height;

        fbo(int width, int height);
        void bind() const;
        static void bind_screen();
        void bind_texture() const;
        static  void unbind_texture();
        ~fbo();
    };
} // gl3

#endif //FBO_H
