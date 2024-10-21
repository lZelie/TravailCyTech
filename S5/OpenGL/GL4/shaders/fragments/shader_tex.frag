#version 430

in vec2 UV;
out vec4 fColor;
layout (location = 1) uniform float time;

void main() {

    fColor = vec4(UV, 0.5 + 0.5 * sin(time), 1.0);
}
