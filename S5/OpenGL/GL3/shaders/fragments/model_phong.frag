#version 430

in vec3 point;
in vec3 normal_vector;
in vec2 tex_coord;
uniform sampler2D tex;

layout (location = 6) uniform vec4 Ka;
layout (location = 7) uniform vec4 Kd;
layout (location = 8) uniform vec4 Ks;
layout (location = 9) uniform float shininess;
layout (location = 10) uniform vec3 lightPos;
layout (location = 11) uniform vec3 viewPos;

out vec4 fColor;

void main() {
    vec3 L = normalize(lightPos - point);
    vec3 V = normalize(viewPos - point);
    vec3 R = reflect(-L, normal_vector);
    vec4 lightColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    vec4 amb = Ka * lightColor;
    vec4 diff = Kd * lightColor * max(dot(normal_vector, L), 0);
    vec4 spec = Ks * lightColor * (dot(normal_vector, L) > 0 ? 1 : 0) * (pow(max(dot(R, V), 0), shininess));

    fColor = amb + diff + spec;
    fColor = clamp(fColor, 0.0, 1.0) * vec4(abs(normal_vector), 1.0f);
}