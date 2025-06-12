import moderngl
import glfw
from OpenGL import GL
from PIL import Image
import os
from typing import List, Tuple, Callable, Optional
from enum import Enum
from dataclasses import dataclass


# Did you know? Python is a garbage programming language. Always was and always will be.
# I hate it with a passion, and it destroys my soul that I have to work with it.

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

def read_file(file_name: str, folder: Optional[str]=None) -> str:
    if folder:
        file_path = os.path.join(folder, file_name)
    else:
        file_path = file_name

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
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


@dataclass(frozen=True)
class ShaderInfo:
    vert_module: str
    frag_module: str
    uniform_types: List[Tuple[str, UniformType]]

    @staticmethod
    def load_from(
            vert_path: str, frag_path: str, folder: str, uniform_types: List[Tuple[str, UniformType]]) -> 'ShaderInfo':
        return ShaderInfo(
            vert_module=read_file(vert_path, folder),
            frag_module=read_file(frag_path, folder),
            uniform_types=uniform_types
        )

    def generate_image(self, width: int, height: int, frag_uniforms=None, attributes=None):
        ctx = moderngl.create_standalone_context()
        prog = ctx.program(
            vertex_shader=self.vert_module,
            fragment_shader=self.frag_module,
        )

        if frag_uniforms is not None:
            for name, value in frag_uniforms.items():
                prog[name].value = value

        if attributes and len(attributes) > 0:
            # Assume 'attributes' is a dict: {attr_name: numpy_array}
            buffers = []
            for attr_name, array in attributes.items():
                buffer = ctx.buffer(array.astype('f4').tobytes())
                buffers.append((buffer, attr_name))
            vao = ctx.vertex_array(prog, buffers)
        else:
            # No attributes used — gl_VertexID assumed
            vao = ctx.vertex_array(prog, [])

        fbo = ctx.simple_framebuffer((width, height))
        fbo.use()
        ctx.clear(0, 0, 0, 1)  # clear to black
        vao.render(moderngl.TRIANGLES, vertices=3)

        data = fbo.read(components=3)
        image = Image.frombytes('RGB', (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image


class GlWindow:

    """
    Generate a window containing an OpenGL shader.
    """
    def __init__(
            self, shader_info: ShaderInfo ,
            width: int, height: int,
            title: str, visible: bool = True
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

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        # Compile and link shader modules
        self.shader_info = shader_info
        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, compile_shader(GL.GL_VERTEX_SHADER, shader_info.vert_module))
        GL.glAttachShader(self.program, compile_shader(GL.GL_FRAGMENT_SHADER, shader_info.frag_module))
        GL.glLinkProgram(self.program)

        self.local_uniforms_setters = {
            uniform_name: (lambda value, uname=uniform_name, utype=uniform_type:
                           uniform_setters[utype](GL.glGetUniformLocation(self.program, uname), value))
            for uniform_name, uniform_type in shader_info.uniform_types
        }

        self.mouse_events = MouseEvents(self)

    def render_start(self):
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        GL.glViewport(0, 0, fb_width, fb_height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self._vao)

    def set_uniform(self, name: str, value):
        self.local_uniforms_setters[name](value)

    def render_complete(self):
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(self.window)

    def draw_screen(self, uniforms = None):
        self.render_start()

        if uniforms is not None:
            for name, value in uniforms.items():
                self.set_uniform(name, value)

        self.render_complete()

    def get_uniform_location(self, name: str):
        return GL.glGetUniformLocation(self.program, name)

    def generate_image(self):
        """
        Note that the size of the generated image is the size of the frame buffer, and not the pixel size of the image.
        """
        target_width, target_height = glfw.get_framebuffer_size(self.window)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, target_width, target_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (target_width, target_height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

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

    def save_image(self, image, extension: str = 'jpg'):
        path = os.path.join(self.folder, f'{self.name}_{self.index}.{extension}')
        image.save(path)
        print(f"Saved to {path}")
        self.index += 1

