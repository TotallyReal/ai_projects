import glfw
from OpenGL import GL
from PIL import Image
import os
from typing import List, Tuple, Callable
from enum import Enum

"""
A simple wrapper file to create shader windows, e.g.

app_window = GlWindow(
    width, height,
    "Title", "shader.vert", "shader.frag",
    [
        ('u_resolution', UniformType.VEC2),
        ('u_dimension',  UniformType.VEC2),
        ('u_position',   UniformType.VEC2),
        ('u_time',       UniformType.FLOAT),
        ...
    ]
)
"""

def load_from_file(path: str):
    with open(path, 'r') as f:
        return f.read()

def compile_shader(shader_type, source):
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
        raise RuntimeError(GL.glGetShaderInfoLog(shader).decode())
    return shader

class UniformType(Enum):
    FLOAT = 1
    INT = 2
    BOOL = 3
    VEC2 = 4
    VEC3 = 5
    VEC4 = 6
    IVEC2 = 7
    IVEC3 = 8
    IVEC4 = 9
    MAT2 = 10
    MAT3 = 11
    MAT4 = 12
    SAMPLER2D = 13


uniform_setters = {
    UniformType.FLOAT: lambda loc, val: GL.glUniform1f(loc, val),
    UniformType.INT:   lambda loc, val: GL.glUniform1i(loc, val),
    UniformType.BOOL:  lambda loc, val: GL.glUniform1i(loc, int(val)),

    UniformType.VEC2:  lambda loc, val: GL.glUniform2f(loc, *val),
    UniformType.VEC3:  lambda loc, val: GL.glUniform3f(loc, *val),
    UniformType.VEC4:  lambda loc, val: GL.glUniform4f(loc, *val),

    UniformType.IVEC2: lambda loc, val: GL.glUniform2i(loc, *val),
    UniformType.IVEC3: lambda loc, val: GL.glUniform3i(loc, *val),
    UniformType.IVEC4: lambda loc, val: GL.glUniform4i(loc, *val),

    UniformType.MAT2:  lambda loc, val: GL.glUniformMatrix2fv(loc, 1, GL.GL_FALSE, val),
    UniformType.MAT3:  lambda loc, val: GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, val),
    UniformType.MAT4:  lambda loc, val: GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, val),

    UniformType.SAMPLER2D: lambda loc, val: GL.glUniform1i(loc, val),
}


class GlWindow:

    """
    Generate a window containing an OpenGL shader.
    """
    def __init__(
            self, width: int, height: int,
            title: str, vert_path: str, frag_path: str,
            uniforms: List[Tuple[str, UniformType]],
            visible: bool = True
    ):
        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        # Set OpenGL context to version 3.3 core, required on macOS for GLSL 330+
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # macOS requires this hint for core profile contexts
        if os.uname().sysname == 'Darwin':
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        self.width = width
        self.height = height

        glfw.make_context_current(self.window)
        print("OpenGL Version:", GL.glGetString(GL.GL_VERSION).decode())
        print("GLSL Version:", GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode())

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Compile and link shaders
        vs = compile_shader(GL.GL_VERTEX_SHADER, load_from_file(vert_path))
        fs = compile_shader(GL.GL_FRAGMENT_SHADER, load_from_file(frag_path))

        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, vs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)

        self.local_uniforms_setters = {
            uniform_name: (lambda value, uname=uniform_name, utype=uniform_type:
                           uniform_setters[utype](GL.glGetUniformLocation(self.program, uname), value))
            for uniform_name, uniform_type in uniforms
        }

        self.mouse_events = MouseEvents(self)

    def render_start(self):
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, fb_width, fb_height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vao)

    def set_uniform(self, name: str, value):
        self.local_uniforms_setters[name](value)

    def render_complete(self):
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(self.window)

    def draw_screen(self, uniforms):
        self.render_start()

        for name, value in uniforms.items():
            self.set_uniform(name, value)

        self.render_complete()

    def get_uniform_location(self, name: str):
        return GL.glGetUniformLocation(self.program, name)

    def generate_image(self):
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, fb_width, fb_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (fb_width, fb_height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def set_scroll_callback(self, scroll_callback):
        glfw.set_scroll_callback(self.window, scroll_callback)

# <editor-fold desc="Events">

def _invoke(listeners, *parameters):
    for callback in listeners:
        callback(*parameters)

# Parameters: GLWindow, x, y
MouseButtonCallback = Callable[[GlWindow, float, float], None]

# Parameters: GLWindow, x, y, dx, dy
MouseDraggedCallback = Callable[[GlWindow, float, float, float, float], None]

# Parameters: GLWindow, x, y, scroll_value
MouseScrollCallback = Callable[[GlWindow, float, float, float], None]

class MouseEvents:
    """
    Support mouse event for the GL_Window, by adding listeners to:
        mouse_pressed_listeners   (MouseButtonCallback)
        mouse_released_listeners  (MouseButtonCallback)
        mouse_dragged_listeners   (MouseDraggedCallback)
        mouse_moved_listeners     (MouseDraggedCallback)
        mouse_scrolled_listeners  (MouseScrollCallback)

    Also, the press status is saved in `is_pressed`, and position in `position()`.
    """

    def __init__(self, gl_window: GlWindow):
        assert gl_window is not None

        self.gl_window = gl_window
        self.window = gl_window.window
        self.is_pressed = False

        self.mouse_pressed_listeners: List[MouseButtonCallback] = []
        self.mouse_released_listeners: List[MouseButtonCallback] = []
        self.last_position = (0, 0)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)

        self.mouse_dragged_listeners: List[MouseDraggedCallback] = []
        self.mouse_moved_listeners: List[MouseDraggedCallback] = []
        glfw.set_cursor_pos_callback(self.window, self._mouse_moved)

        self.mouse_scrolled_listeners: List[MouseScrollCallback] = []
        glfw.set_scroll_callback(self.window, self._mouse_scrolled)

    def position(self):
        return glfw.get_cursor_pos(self.window)

    def _mouse_button(self, window: glfw._GLFWwindow, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return

        x, y = self.position()

        if action == glfw.PRESS:
            self.is_pressed = True
            _invoke(self.mouse_pressed_listeners, self.gl_window, x, y)
        elif action == glfw.RELEASE:
            self.is_pressed = False
            _invoke(self.mouse_released_listeners, self.gl_window, x, y)

        self.last_position = (x,y)

    def _mouse_moved(self, window: glfw._GLFWwindow, x: float, y: float):

        x, y = self.position()
        dx, dy = x - self.last_position[0], y - self.last_position[1]

        if self.is_pressed:
            _invoke(self.mouse_dragged_listeners, self.gl_window, x, y, dx, dy)
        else:
            _invoke(self.mouse_moved_listeners, self.gl_window, x, y, dx, dy)

        self.last_position = (x,y)

    def _mouse_scrolled(self, window: glfw._GLFWwindow, x_scroll: float, y_scroll: float):
        mouse_x, mouse_y = self.position()
        _invoke(self.mouse_scrolled_listeners, self.gl_window, mouse_x, mouse_y, y_scroll)

# </editor-fold>

# TODO: Move to another file

class IndexedSaver:
    """
    Used to save images with the same name, and running index.
    """

    def __init__(self, folder: str, name: str):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.name = name
        self.index = 0

    def save_image(self, image):
        path = os.path.join(self.folder, f'{self.name}_{self.index}.jpg')
        image.save(path)
        print(f"Saved to {path}")
        self.index += 1

