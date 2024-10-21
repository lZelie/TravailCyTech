#version 430
layout (location = 0) in vec4 v_coord;
layout (location = 1) in vec2 v_tex_coord;
layout (location = 2) in vec3 v_normal;

layout (location = 3) uniform mat4 M;
layout (location = 4) uniform mat4 V;
layout (location = 5) uniform mat4 P;

out vec3 point;
out vec3 normal_vector;
out vec2 tex_coord;

void main(){
    tex_coord = v_tex_coord;

    point = (V * M * v_coord).xyz;
    normal_vector = normalize(transpose(inverse(mat3(V))) * v_normal);

    gl_Position = P * V * M * v_coord;
}
