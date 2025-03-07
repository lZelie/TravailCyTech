//
// Created by VISUAL on 07/03/2025.
//

#ifndef RENDERER_H
#define RENDERER_H
#include <memory>
#include <vector>

#include "camera.h"
#include "vao.h"
#include "GLFW/glfw3.h"

// Application constants
constexpr int INITIAL_WIDTH = 640;
constexpr int INITIAL_HEIGHT = 480;
constexpr float CAMERA_FOV = 45.0f;
constexpr char WINDOW_TITLE[] = "Hello World!";

// Scene data
constexpr std::array<glm::vec4, 256> SPHERES = {
    glm::vec4{0.0f, 0.0f, -5.0f, 1.0f}, 
    {1.0f, 1.0f, 1.0f, 0.1f}, 
    {15.0f, 10.0f, 15.0f, 10.0f}
};

constexpr std::array<glm::vec3, 256> PLANES = {
    glm::vec3{0.0f, -1.0f, 0.0f}, 
    glm::vec3{0.0f, 1.0f, 0.0f}
};

constexpr std::array<glm::vec3, 256> TRIANGLES = {
    // Pyramid
    glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
};

// Lighting
constexpr glm::vec4 LIGHT_POSITION{100.0f, 100.0f, -100.0f, 0.9f};
constexpr glm::vec3 LIGHT_COLOR{1.0f, 1.0f, 0.99f};
constexpr glm::vec3 LIGHT_AMBIENT{0.1f, 0.1f, 0.1f};

// Shader uniform locations
struct ShaderLocations {
    static constexpr int NUM_OBJECTS = 3;
    static constexpr int NUM_PLANES = 1;
    static constexpr int NUM_TRIANGLES = 4;
    
    static constexpr int VIEW_DATA = 3;
    static constexpr int SPHERES_DATA = 19;
    static constexpr int PLANES_DATA = 276;
    static constexpr int TRIANGLES_DATA = 534;
    static constexpr int LIGHT_POSITION = 800;
    static constexpr int LIGHT_COLOR = 801;
    static constexpr int LIGHT_AMBIENT = 802;
    static constexpr int LIGHT_TYPE = 803;
};


class Renderer {
private:
    GLFWwindow* window;
    std::vector<float> windowSize;
    float cameraFov;
    gl3::camera camera;
    int light_type;
    std::unique_ptr<gl3::shader_class> shaderProgram;
    std::unique_ptr<gl3::vao> quadVAO;
    std::unique_ptr<gl3::vbo> quadVBO;

    // Initialize GLFW and create window
    void initWindow();

    // Set up callback functions
    void setupCallbacks();

    // Initialize quad for rendering
    void initQuad();

    // Set up view data for shader
    [[nodiscard]] std::vector<float> prepareViewData() const;

    // Send data to shader uniforms
    void updateShaderUniforms() const;

public:
    Renderer();
    
    ~Renderer();

    void renderFrame();

    [[nodiscard]] bool shouldClose() const;
};




#endif //RENDERER_H
