//
// Created by zelie on 10/7/24.
//

#include "texture.h"

#include <vector>
#include <fstream>
#include <iostream>

#include "app.h"

namespace gl3 {

    std::string load_texture_bmp(const char* filename, const int width, const int height)
    {
        std::ifstream file{filename, std::ios::in};
        std::string text;
        file >> text;
        for (int i = 0; i < width * height; i++)
        {
            const int index = i * 3;
            const unsigned char b = text[index];
            const unsigned char r = text[index + 2];
            text[index] = r;
            text[index + 2] = b;
        }
        return text;
    }

    std::vector<float> generate_checkers(const int width, const int height)
    {
        std::vector<float> data(width * height * 4);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                const int c = (i & 0x8) == 0 ^ (j & 0x8) == 0;
                data[4 * (i * height + j)] = c;
                data[4 * (i * height + j) + 1] = c;
                data[4 * (i * height + j) + 2] = c;
                data[4 * (i * height + j) + 3] = 1.0f;
            }
        }
        return data;
    }

    texture::texture()
    {
        const std::vector white{1.0f, 1.0f, 1.0f};
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGBA, GL_FLOAT, white.data());
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    texture::texture(const int width, const int height)
    {
        const auto data = generate_checkers(width, height);
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, data.data());
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    texture::texture(const char* image, const int width, const int height)
    {
        const auto data = load_texture_bmp(image, width, height);
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    void texture::tex_unit(shader_class& shader, const char* uniform, const GLuint unit)
    {
        const GLuint tex_uni = glGetUniformLocation(shader.ID, uniform);
        shader.activate();
        glUniform1i(tex_uni, unit);
    }


    void texture::bind() const
    {
        glBindTexture(GL_TEXTURE_2D, id);
    }

    void texture::unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    texture::~texture()
    {
        glDeleteTextures(1, &id);
    }
} // gl3