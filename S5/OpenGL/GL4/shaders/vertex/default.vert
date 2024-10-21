#version 430 core

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;

layout (location = 2) uniform mat4 MVP;

out vec4 color;

void main() {
    color = vColor;
    gl_Position = MVP * vPosition;
}