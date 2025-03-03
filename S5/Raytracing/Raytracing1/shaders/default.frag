#version 430

layout (location = 0) uniform float view[16]; // send SCREEN_X and SCREEN_Y
layout (location = 1) uniform float sphere[16];

vec2 UV; // the UV coordinates of this pixel on the canvas

out vec4 fColor; // final color

vec2 compute_uv() {
    // Get the aspect ratio (width / height)
    float aspect = view[0] / view[1];

    // Calculate normalized device coordinates (NDC) in the range [-1, 1]
    vec2 ndc = (gl_FragCoord.xy / vec2(view[0], view[1]) - 0.5) * 2.0;

    // Scale coordinates so the shorter dimension has length 1 and
    // aspect ratio is maintained, ensuring a perfect circle
    if (aspect >= 1.0) {
        // Width is larger, scale X by 1/aspect ratio
        return vec2(ndc.x * aspect, ndc.y);
    } else {
        // Height is larger, scale Y by aspect ratio
        return vec2(ndc.x, ndc.y / aspect);
    }
}

void main() {
    UV = compute_uv();
    vec3 color = vec3(0.0f, 0.0f, 0.0f);
    if (length(UV) < 1){
        color = vec3(1.0f, 0.0f, 0.0f);
    }
    fColor = vec4(color, 1.0f);
}