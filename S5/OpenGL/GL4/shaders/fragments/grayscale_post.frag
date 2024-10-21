#version 430

in vec2 UV;
uniform sampler2D tex;
out vec4 fColor;

void main(){
    vec4 color = texture(tex, UV);

    float grayscale = (color.r + color.g + color.b) / 3.0;

    fColor = vec4(grayscale, grayscale, grayscale, color.a);
}