#version 430

in vec2 UV;
layout(location = 1) uniform float pixel_size;
uniform sampler2D tex;
out vec4 fColor;

void main(){
    vec2 pixel_coords = floor(UV / pixel_size) * pixel_size;

    fColor = texture(tex, pixel_coords);
}