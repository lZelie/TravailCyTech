//
// Created by VISUAL on 07/03/2025.
//

#include "Renderer.h"

#include <iostream>

void Renderer::initWindow()
{
    glfwInit();

    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create window
    window = glfwCreateWindow(static_cast<int>(windowSize[0]), static_cast<int>(windowSize[1]), WINDOW_TITLE, nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // Set callbacks
    setupCallbacks();

    // Initialize GLEW
    if (const GLenum err = glewInit(); err != GLEW_OK)
    {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Renderer::setupCallbacks()
{
    // Window resize callback
    glfwSetWindowSizeCallback(window, [](GLFWwindow* win, int width, int height)
    {
        const auto renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(win));
        renderer->windowSize[0] = static_cast<float>(width);
        renderer->windowSize[1] = static_cast<float>(height);
    });

    // Keyboard input callback
    glfwSetKeyCallback(window, [](GLFWwindow* win, const int key, int, const int action, int)
    {
        const auto renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(win));

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }
        if (key == GLFW_KEY_1 && action == GLFW_PRESS)
        {
            renderer->light_type = renderer->light_type == 0 ? 1 : 0;
        }
    });

    // Store this instance for callbacks
    glfwSetWindowUserPointer(window, this);
}

void Renderer::initQuad()
{
    const std::vector vertices = {
        -1.0f, 1.0f, // Top-left
        -1.0f, -1.0f, // Bottom-left
        1.0f, 1.0f, // Top-right
        1.0f, -1.0f // Bottom-right
    };

    quadVAO->bind();
    quadVBO = std::make_unique<gl3::vbo>(vertices);
    quadVAO->linkAttrib(*quadVBO, 0, 2, GL_FLOAT, 0, nullptr);
}

std::vector<float> Renderer::prepareViewData() const
{
    return {
        windowSize[0], windowSize[1],
        camera.Position[0], camera.Position[1], camera.Position[2],
        camera.Orientation[0] + camera.Position[0],
        camera.Orientation[1] + camera.Position[1],
        camera.Orientation[2] + camera.Position[2],
        cameraFov,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
}

void Renderer::updateShaderUniforms() const
{
    const std::vector view = prepareViewData();

    // Set object counts
    glUniform1i(0, ShaderLocations::NUM_OBJECTS);
    glUniform1i(1, ShaderLocations::NUM_PLANES);
    glUniform1i(2, ShaderLocations::NUM_TRIANGLES);

    // Set view data
    glUniform1fv(ShaderLocations::VIEW_DATA, static_cast<int>(view.size()), view.data());

    // Set scene geometry
    glUniform4fv(ShaderLocations::SPHERES_DATA, static_cast<int>(SPHERES.size()), &SPHERES[0][0]);
    glUniform3fv(ShaderLocations::PLANES_DATA, static_cast<int>(PLANES.size()), &PLANES[0][0]);
    glUniform3fv(ShaderLocations::TRIANGLES_DATA, static_cast<int>(TRIANGLES.size()), &TRIANGLES[0][0]);

    // Set lighting parameters
    glUniform4fv(ShaderLocations::LIGHT_POSITION, 1, &LIGHT_POSITION[0]);
    glUniform3fv(ShaderLocations::LIGHT_COLOR, 1, &LIGHT_COLOR[0]);
    glUniform3fv(ShaderLocations::LIGHT_AMBIENT, 1, &LIGHT_AMBIENT[0]);
    glUniform1i(ShaderLocations::LIGHT_TYPE, light_type);
    glUniform1i(ShaderLocations::SAMPLE_RATE, SAMPLE_RATE);
}

Renderer::Renderer() :
    window(nullptr),
    windowSize{INITIAL_WIDTH, INITIAL_HEIGHT},
    cameraFov(CAMERA_FOV),
    camera(INITIAL_WIDTH, INITIAL_HEIGHT, {0.0f, 1.0f, 1.0f}),
    light_type(0)
{
    initWindow();
    shaderProgram = std::make_unique<gl3::shader_class>("shaders/default.vert", "shaders/default.frag");
    quadVAO = std::make_unique<gl3::vao>();
    initQuad();

    // Set default clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

Renderer::~Renderer()
{
    glfwTerminate();
}

void Renderer::renderFrame()
{
    // Update viewport and clear buffers
    glViewport(0, 0, static_cast<int>(windowSize[0]), static_cast<int>(windowSize[1]));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate shader and update uniforms
    shaderProgram->activate();
    updateShaderUniforms();

    // Draw quad
    quadVAO->bind();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Swap buffers and process events
    glfwSwapBuffers(window);
    glfwPollEvents();

    // Process camera inputs
    camera.Inputs(window);
}

bool Renderer::shouldClose() const
{
    return glfwWindowShouldClose(window);
}
