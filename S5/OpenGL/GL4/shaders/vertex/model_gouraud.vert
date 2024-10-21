#version 430
layout (location = 0) in vec4 v_coord;
layout (location = 1) in vec2 v_tex_coord;
layout (location = 2) in vec3 v_normal;

layout (location = 3) uniform mat4 M;
layout (location = 4) uniform mat4 V;
layout (location = 5) uniform mat4 P;
layout (location = 6) uniform vec4 Ka;
layout (location = 7) uniform vec4 Kd;
layout (location = 8) uniform vec4 Ks;
layout (location = 9) uniform float shininess;
layout (location = 10) uniform vec3 lightPos;
layout (location = 11) uniform vec3 viewPos;


out vec4 color;
out vec2 tex_coord;

void main(){
    vec3 N = normalize(transpose(inverse(mat3(V * M))) * v_normal);
    tex_coord = v_tex_coord;

    gl_Position = P * V * M * v_coord;

    vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);

    vec4 amb = Ka * lightColor;

    vec3 L = normalize(lightPos - (M * v_coord).xyz);

    vec4 diff = Kd * lightColor * max(dot(v_normal, L), 0);

    vec3 V = normalize(viewPos - (M * v_coord).xyz);

    vec3 R = reflect(-normalize(lightPos - (M * v_coord).xyz), v_normal);

    vec4 spec = Ks * lightColor * (dot(v_normal, L) > 0 ? 1 : 0) * (pow(max(dot(R, V), 0), shininess));

    color = amb + diff + spec;


    color = clamp(color, 0.0, 1.0) * vec4(abs(N), 1.0f);
}
