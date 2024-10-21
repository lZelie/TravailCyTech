#version 430

layout (location = 0) in vec2 in_position;
out vec2 UV;

void main() {
    UV = (1.0f + in_position) / 2.0f;
    gl_Position = vec4(in_position, 0.0f, 1.0f);
}