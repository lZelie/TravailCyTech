#version 430

in vec4 color;
in vec2 tex_coord;
uniform sampler2D tex;

out vec4 fColor;

void main() {
    fColor = texture(tex, tex_coord);
}