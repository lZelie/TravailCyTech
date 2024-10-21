#version 430
layout (location = 0) in vec4 v_coord;
layout (location = 1) in vec2 v_tex_coord;
layout (location = 2) in vec3 v_normal;

layout (location = 3) uniform mat4 M;
layout (location = 4) uniform mat4 V;
layout (location = 5) uniform mat4 P;

out vec4 color;
out vec2 tex_coord;

void main(){
    tex_coord = v_tex_coord;

    vec3 N = normalize(transpose(inverse(mat3(V * M))) * v_normal);
    color = vec4(abs(N), 1.0f);

    gl_Position = P * V * M * v_coord;
}
