//
// Created by cytech on 24/09/24.
//

#include <GL/glew.h>
#include "camera.h"
#include "shader_class.h"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

gl3::camera::camera(int width, int height, glm::vec3 Position): width(width), height(height), Position(Position)
{
}

void gl3::camera::Matrix(float FOVdeg, float nearPlane, float farPlane, gl3::shader_class& shader, const char* uniform)
{
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    view = glm::lookAt(Position, Position + Orientation, Up);
    projection = glm::perspective(glm::radians(FOVdeg), (float)(width / height), nearPlane, farPlane);

    glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(projection * view));
}

void gl3::camera::Inputs(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        Position += speed * Orientation;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        Position += speed * -glm::normalize(glm::cross(Orientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        Position += speed * -Orientation;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        Position += speed * glm::normalize(glm::cross(Orientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        Position += speed * Up;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        Position += speed * -Up;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        speed = 0.004f;
    }
    else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
    {
        speed = 0.001f;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // DISABLED is better than HIDDEN

        if (firstClick)
        {
            glfwSetCursorPos(window, (width / 2), (height / 2));
            firstClick = false;
        }

        double mX, mY;
        glfwGetCursorPos(window, &mX, &mY);

        // Calculate rotation based on mouse movement from center
        float rotX = sensitivity * (float)(mY - (height / 2)) / height;
        float rotY = sensitivity * (float)(mX - (width / 2)) / width;  // Fixed: use width here

        // Calculate new orientation with pitch (X) rotation
        glm::vec3 right = glm::normalize(glm::cross(Orientation, Up));
        glm::vec3 newOrientation = glm::rotate(Orientation, glm::radians(-rotX), right);

        // Prevent camera from flipping by checking the angle with up vector
        // Allow more range (10-15 degrees) for natural movement
        if (!(glm::angle(newOrientation, Up) <= glm::radians(10.0f) ||
              glm::angle(newOrientation, -Up) <= glm::radians(10.0f)))
        {
            Orientation = newOrientation;
        }

        // Apply yaw (Y) rotation around the Up axis
        Orientation = glm::normalize(glm::rotate(Orientation, glm::radians(-rotY), Up));

        // Reset cursor position to center
        glfwSetCursorPos(window, (width / 2), (height / 2));
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstClick = true;
    }
}

void gl3::camera::MoveKey(GLFWwindow*, const int key, int, const int action, int)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        switch (key)
        {
        case GLFW_KEY_W:
            {
                Position += speed * Orientation;
                break;
            }
        case GLFW_KEY_A:
            {
                Position += speed * -glm::normalize(glm::cross(Orientation, Up));
                break;
            }
        case GLFW_KEY_S:
            {
                Position += speed * -Orientation;
                break;
            }
        case GLFW_KEY_D:
            {
                Position += speed * glm::normalize(glm::cross(Orientation, Up));
                break;
            }
        case GLFW_KEY_SPACE:
            {
                Position += speed * Up;
                break;
            }
        case GLFW_KEY_LEFT_CONTROL:
            {
                Position += speed * -Up;
                break;
            }
        default: break;
        }
    }
    if (key == GLFW_KEY_LEFT_SHIFT)
    {
        if (action == GLFW_PRESS)
        {
            speed = 4.0f;
        }
        else if (action == GLFW_RELEASE)
        {
            speed = 1.0f;
        }
    }
}
