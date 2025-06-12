#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 u_frame_buffer_resolution;   // (window width, height in pixels)
uniform vec2 u_resolution;                // (framebuffer width, height in pixels)
uniform float u_radius;                   // circle radius in pixels

void main() {
    // gl_FragCoord is coordinates in the framebuffer, not in pixels!
    // The following is in [0,1]x[0,1] as relative position in window.
    // This is the same as the computation of uv in the vert part
    vec2 normalizePos = gl_FragCoord.xy/u_frame_buffer_resolution.xy;

    // UV gradient color
    vec3 color = vec3(normalizePos, 0.0);

    // Create a blue ball in an ABSOLUTE position in pixel and radius.
    // Will not change if we resize the window
    vec2 pixelPos = normalizePos * u_resolution;
    vec2 center = vec2(100.0,100.0);

    if (length(pixelPos - center) < u_radius){
        color = vec3(0,0,1.0);
    }

    // Create a white ball in a relative position (center of screen) with radius 100 (in bufferframe)
    if (length(gl_FragCoord.xy - 0.5*u_frame_buffer_resolution) < u_radius){
        color = vec3(1);
    }


    fragColor = vec4(color, 1.0);
}
