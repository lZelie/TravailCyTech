//
// Created by zelie on 10/10/24.
//

#include "fbo.h"

#include <iostream>

#include "app.h"

namespace gl3
{
    fbo::fbo(const int width, const int height): width(width), height(height)
    {
        glGenFramebuffers(1, &id);
        glBindFramebuffer(GL_FRAMEBUFFER, id);

        glGenTextures(1, &tex_id);
        glBindTexture(GL_TEXTURE_2D, tex_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex_id, 0);
        const std::vector<GLenum> draw_buffers = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, draw_buffers.data());

        glGenRenderbuffers(1, &depth_render_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_render_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_render_buffer);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void fbo::bind() const
    {
        glBindFramebuffer(GL_FRAMEBUFFER, id);
    }

    void fbo::bind_screen()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void fbo::bind_texture() const
    {
        glBindTexture(GL_TEXTURE_2D, tex_id);
    }

    void fbo::unbind_texture()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    fbo::~fbo()
    {
        glDeleteFramebuffers(1, &id);
        glDeleteTextures(1, &tex_id);
        glDeleteRenderbuffers(1, &depth_render_buffer);
    }
} // gl3
