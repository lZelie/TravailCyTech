#version 430

in vec2 UV;
layout (location = 1) uniform float radius;
uniform sampler2D tex;
out vec4 fColor;

void main(){
    vec4 sum = vec4(0.0);
    float total = 0.0;

    for (int x = -int(radius); x<= int(radius); ++x) {
        for (int y = -int(radius); y <= int(radius); ++y) {
            vec2 offset = vec2(float(x), float(y)) / textureSize(tex, 0);
            sum += texture(tex, UV + offset);
            total += 1.0;
        }
    }
    fColor = sum / total;
}