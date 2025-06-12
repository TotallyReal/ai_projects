import glfw

from attention.shaders.shader import GlWindow, UniformType, IndexedSaver

width = 512
height = 256

app_window = GlWindow(
    width, height,
    "Circle uv", "circle_uv.vert", "circle_uv.frag",
    [
        ('u_resolution',              UniformType.VEC2),
        ('u_frame_buffer_resolution', UniformType.VEC2),
        ('u_radius',                  UniformType.FLOAT),
    ]
)

uniforms = {
    'u_resolution': [width, height],
    'u_frame_buffer_resolution': [width, height],
    'u_radius'    : 10,
}


while not glfw.window_should_close(app_window.window):
    glfw.poll_events()
    uniforms['u_frame_buffer_resolution'] = glfw.get_framebuffer_size(app_window.window)
    uniforms['u_resolution'] = glfw.get_window_size(app_window.window)
    app_window.draw_screen(uniforms)
