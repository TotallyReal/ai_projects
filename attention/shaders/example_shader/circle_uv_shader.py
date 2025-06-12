import glfw
import os

from attention.shaders.shader import GlWindow, UniformType, ShaderInfo

width = 512
height = 256

shader_info = ShaderInfo.load_from(
    vert_path='circle_uv.vert', frag_path='circle_uv.frag', folder=os.path.dirname(os.path.abspath(__file__)),
    uniform_types=[
        ('u_resolution',              UniformType.VEC2),
        ('u_frame_buffer_resolution', UniformType.VEC2),
        ('u_radius',                  UniformType.FLOAT),
    ]
)

app_window = GlWindow(
    shader_info=shader_info,
    width=width, height=height,
    title="Circle uv"
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
