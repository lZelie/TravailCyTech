#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "camera.h"
#include "shader_class.h"
#include "vao.h"

std::vector<float> window_size = {640, 480};
glm::vec3 camera_from = glm::vec3(0.0f, 1.0f, 1.0f);
glm::vec3 camera_to = glm::vec3(0.0f, 0.0f, -1.0f);
float camera_fov = 45.0f;
gl3::camera camera(640, 480, {0.0f, 1.0f, 1.0f});
int light = 0;

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
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (key == GLFW_KEY_1 && action == GLFW_PRESS)
        {
            light = light == 0 ? 1 : 0;
        }
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

    constexpr std::array<glm::vec4, 256> spheres = {glm::vec4{0.0f, 0.0f, -5.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.1f}, {15.0f, 10.0f, 15.0f, 10.0f}};
    constexpr std::array<glm::vec3, 256> planes = {glm::vec3{0.0f, -1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}};
    constexpr std::array<glm::vec3, 256> triangles = {
        glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
        glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
        glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
        glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    };
    constexpr glm::vec4 light_pos{15.0f, 1.0f, -15.0f, 100.0f};
    constexpr glm::vec3 light_color{1.0f, 1.0f, 1.0f};
    constexpr glm::vec3 light_ambient{0.1f, 0.1f, 0.1f};
    while (!glfwWindowShouldClose(window))
    {
        glViewport(0, 0, window_size[0], window_size[1]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        std::vector view = {
            window_size[0], window_size[1], camera.Position[0], camera.Position[1], camera.Position[2],
            camera.Orientation[0] + camera.Position[0], camera.Orientation[1] + camera.Position[1], camera.Orientation[2] + camera.Position[2], camera_fov, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        };
        glUniform1i(0, 3);
        glUniform1i(1, 1);
        glUniform1i(2, 4);
        glUniform1fv(3, 16, view.data());
        glUniform4fv(19, 256, &spheres[0][0]);
        glUniform3fv(276, 256, &planes[0][0]);
        glUniform3fv(534, 256, &triangles[0][0]);
        glUniform4fv(800, 1, &light_pos[0]);
        glUniform3fv(801, 1, &light_color[0]);
        glUniform3fv(802, 1, &light_ambient[0]);
        glUniform1i(803, light);
        program_default.activate();
        vao_quad.bind();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
        camera.Inputs(window);
    }

    return 0;
}
