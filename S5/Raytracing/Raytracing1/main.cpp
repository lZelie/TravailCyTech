#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "shader_class.h"
#include "vao.h"

std::vector<float> window_size = {640, 480};


int main()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    auto window = glfwCreateWindow(640, 480, "Hello World!", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback(window, [](GLFWwindow*, int width, int height)
    {
        window_size[0] = static_cast<float>(width);
        window_size[1] = static_cast<float>(height);
    });

    if (const GLenum err = glewInit(); err != GLEW_OK)
    {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    const gl3::shader_class program_default("shaders/default.vert", "shaders/default.frag");

    // init quad
    const std::vector vertices = {
        -1.0f, 1.0f,
        -1.0f, -1.0f,
        1.0f, 1.0f,
        1.0f, -1.0f
    };
    gl3::vao vao_quad{};
    vao_quad.bind();
    gl3::vbo vbo_quad(vertices);
    vao_quad.linkAttrib(vbo_quad, 0, 2, GL_FLOAT, 0, nullptr);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    while (!glfwWindowShouldClose(window))
    {
        glViewport(0, 0, window_size[0], window_size[1]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniform1fv(0, 2, window_size.data());
        program_default.activate();
        vao_quad.bind();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    return 0;
}
