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
constexpr char WINDOW_TITLE[] = "RTX on";
constexpr int SAMPLE_RATE = 1;
constexpr int FPS_UPDATE_DELAY = 1;

// Scene data
constexpr std::array<glm::vec4, 256> SPHERES = {
    glm::vec4{5.0f, 0.0f, -10.0f, 1.0f}, 
    {14.0f, 1.0f, -16.0f, 0.1f}, 
    {15.0f, 10.0f, 15.0f, 10.0f}
};

constexpr std::array<glm::vec4, 4> CSG = {
    glm::vec4{-1.0f, 2.0f, 0.0f, 1.5f}, 
    {1.0f, 2.0f, 0.0f, 1.5f}, 
    {0.0f, 2.7f, -0.3f, 0.8f},
    {0.0f, 2.8f, 0.3f, 0.8f},
};

constexpr std::array<glm::vec3, 256> PLANES = {
    // Bottom face: normal pointing down (-Y)
    glm::vec3{0.0f, -30.0f, 0.0f},
    glm::vec3{0.0f, 1.0f, 0.0f},
    // Top face: normal pointing up (+Y)
    glm::vec3{0.0f, 30.0f, 0.0f},
    glm::vec3{0.0f, -1.0f, 0.0f},
    // Left face: normal pointing left (-X)
    glm::vec3{-30.0f, 0.0f, 0.0f},
    glm::vec3{1.0f, 0.0f, 0.0f},
    // Right face: normal pointing right (+X)
    glm::vec3{30.0f, 0.0f, 0.0f},
    glm::vec3{-1.0f, 0.0f, 0.0f},
    // Back face: normal pointing backward (-Z)
    glm::vec3{0.0f, 0.0f, -30.0f},
    glm::vec3{0.0f, 0.0f, 1.0f},
    // Front face: normal pointing forward (+Z)
    glm::vec3{0.0f, 0.0f, 30.0f},
    glm::vec3{0.0f, 0.0f, -1.0f},
};

constexpr std::array<glm::vec3, 256> TRIANGLES = {
    // Pyramid
    glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{3.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{5.0f, -1.0f, 5.0f}, glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
    glm::vec3{5.0f, -1.0f, 3.0f}, glm::vec3{3.0f, -1.0f, 3.0f}, glm::vec3{4.0f, 2.0f, 4.0f},
};

// Lighting
constexpr glm::vec4 LIGHT_POSITION{15.0f, 10.0f, -15.0f, 0.9f};
constexpr glm::vec3 LIGHT_COLOR{1.0f, 1.0f, 0.99f};
constexpr glm::vec3 LIGHT_AMBIENT{0.1f, 0.1f, 0.1f};

// Shader uniform locations
struct ShaderLocations {
    static constexpr int NUM_OBJECTS = 3;
    static constexpr int NUM_PLANES = 6;
    static constexpr int NUM_TRIANGLES = 4;
    
    static constexpr int VIEW_DATA = 3;
    static constexpr int SPHERES_DATA = 19;
    static constexpr int PLANES_DATA = 276;
    static constexpr int TRIANGLES_DATA = 534;
    static constexpr int LIGHT_POSITION = 800;
    static constexpr int LIGHT_COLOR = 801;
    static constexpr int LIGHT_AMBIENT = 802;
    static constexpr int LIGHT_TYPE = 803;
    static constexpr int SAMPLE_RATE = 804;
    static constexpr int CSG_SPHERES = 805;
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
    unsigned frame_acc = 0;
    double prev_fps_update = 0;
    double currentFPS = 0;
    void updateFps();
    
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
