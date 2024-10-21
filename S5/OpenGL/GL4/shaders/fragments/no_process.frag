#version 430

in vec2 UV;
uniform sampler2D tex;
out vec4 fColor;

void main(){
    fColor = vec4(texture(tex, UV).xyz, 1.0);
}