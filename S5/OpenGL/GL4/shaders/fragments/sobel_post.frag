#version 430

in vec2 UV;
out vec4 fColor;

uniform sampler2D tex;

const mat3x3 sobelX = mat3x3(
-1.0, 0.0, 1.0,
-2.0, 0.0, 2.0,
-1.0, 0.0, 1.0
);

const mat3x3 sobelY = mat3x3(
-1.0, -2.0, -1.0,
0.0,  0.0,  0.0,
1.0,  2.0,  1.0
);

void main() {
    vec4 sumX = vec4(0.0);
    vec4 sumY = vec4(0.0);

    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(float(x), float(y)) / textureSize(tex, 0);
            vec4 color = texture(tex, UV + offset);
            sumX += color * sobelX[x + 1][y + 1];
            sumY += color * sobelY[x + 1][y + 1];
        }
    }

    vec4 edges = sqrt(sumX * sumX + sumY * sumY);
    fColor = edges;
}