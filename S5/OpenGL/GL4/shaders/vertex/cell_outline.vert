#version 430
layout (location = 0) in vec4 v_coord;
layout (location = 1) in vec2 v_tex_coord;
layout (location = 2) in vec3 v_normal;

layout (location = 3) uniform mat4 M;
layout (location = 4) uniform mat4 V;
layout (location = 5) uniform mat4 P;

void main(){

    vec4 moved_coord = v_coord + vec4(v_normal * 0.05f, 0.0f);

    gl_Position = P * V * M * moved_coord;
}
