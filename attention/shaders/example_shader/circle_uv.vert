#version 330 core

/**
This is the vertex shader for rendering a full-screen shader using a single triangle.

In OpenGL, fragments (pixels) are only shaded inside the bounds of triangles. To cover the full window,
we draw one large triangle that spans the entire screen. This triangle has vertices at (0,0), (2,0), and (0,2),
which fully contains the screen rectangle [0,1] x [0,1].

The vertex shader runs once per vertex (so total of 3 times). Each pixel inside the triangle will be shaded by the
fragment shader, which receives interpolated values based on the triangle's vertex outputs.

The input for the vertex shader is:
- gl_VertexID in {0, 1, 2}, defining the triangle’s vertices.
*/

out vec2 v_uv;

void main() {
    // Compute vertex position in [0,2]^2 using bit tricks for gl_VertexID = 0,1,2
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);  // 0,1,2 → (0,0), (2,0), (0,2)

    // This will be interpolated and passed to the fragment shader as the [0,1]^2 coordinate
    v_uv = pos;

    // Convert from [0,1]^2 to [-1,1]^2
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
