//
// Created by cytech on 24/09/24.
//

#ifndef GL2_CAMERA_H
#define GL2_CAMERA_H


#include <glm/vec3.hpp>
#include "shader_class.h"
#include "GLFW/glfw3.h"

namespace gl3
{
    class camera
    {
    public:
        glm::vec3 Position;
        glm::vec3 Orientation = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);

        bool firstClick = true;

        int width, height;

        float speed = 1.0f, sensitivity = 100.0f;

        camera(int width, int height, glm::vec3 Position);

        void Matrix(float FOVdeg, float nearPlane, float farPlane, gl3::shader_class& shader, const char* uniform);

        void Inputs(GLFWwindow* window);

        void MoveKey(GLFWwindow* window, int key, int scancode, int action, int mods);
    };
}

#endif //GL2_CAMERA_H
