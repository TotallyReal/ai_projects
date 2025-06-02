import glfw
from OpenGL import GL
from PIL import Image
import os
from typing import List, Tuple, Callable
from enum import Enum

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

    def __init__(
            self, width: int, height: int,
            title: str, vert_path: str, frag_path: str,
            uniforms: List[Tuple[str, UniformType]]
    ):
        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        self.width = width
        self.height = height

        glfw.make_context_current(self.window)

        # Compile and link shaders
        vs = compile_shader(GL.GL_VERTEX_SHADER, load_from_file(vert_path))
        fs = compile_shader(GL.GL_FRAGMENT_SHADER, load_from_file(frag_path))

        self.program = GL.glCreateProgram()
        GL.glAttachShader(self.program, vs)
        GL.glAttachShader(self.program, fs)
        GL.glLinkProgram(self.program)

        self.uniform_locations = {}
        self.local_uniforms_setters = {
            uniform_name: (lambda value, uname=uniform_name, utype=uniform_type:
                           uniform_setters[utype](GL.glGetUniformLocation(self.program, uname), value))
            for uniform_name, uniform_type in uniforms
        }

        self.mouse_events = MouseEvents(self)


    def render_start(self):
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.program)

    def set_uniform(self, name: str, value):
        self.local_uniforms_setters[name](value)

    def render_complete(self):
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(self.window)

    def get_uniform_location(self, name: str):
        return GL.glGetUniformLocation(self.program, name)

    def generate_image(self):
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)  # ??? read data from back buffer
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL origin is bottom-left
        return image

    def set_scroll_callback(self, scroll_callback):
        glfw.set_scroll_callback(self.window, scroll_callback)

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



class IndexedSaver:

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

# Parameters
speed = 2.0
amplitude = 0.9
wavelength = 0.15

saver = IndexedSaver('data', 'julia')

def main():
    window1 = GlWindow(
        512, 512,
        "Julia set", "julia.vert", "julia.frag",
        [
            ('u_resolution', UniformType.VEC2),
            ('u_dimension', UniformType.VEC2),
            ('u_position', UniformType.VEC2),
            ('u_constant', UniformType.VEC2),
            ('u_density', UniformType.FLOAT),
            ('u_color_wave', UniformType.FLOAT),
        ]
    )

    start_time = glfw.get_time()

    while not glfw.window_should_close(window1.window):
        glfw.poll_events()
        time = glfw.get_time() - start_time

        # Key handling
        if glfw.get_key(window1.window, glfw.KEY_S) == glfw.PRESS:
            image = window1.generate_image()
            saver.save_image(image)

            glfw.wait_events_timeout(0.2)  # debounce

        # Render
        window1.render_start()

        window1.set_uniform('u_resolution', [512, 512])
        window1.set_uniform('u_dimension', [2.4,2.4])
        window1.set_uniform('u_position', [-1.2,-1.2])
        window1.set_uniform('u_constant', [-0.37,-0.36])
        window1.set_uniform('u_density', 1)
        window1.set_uniform('u_color_wave', 0)


        # window1.set_uniform('iTime', time)
        # window1.set_uniform('speed', speed)
        # window1.set_uniform('amplitude', amplitude)
        # window1.set_uniform('wavelength', wavelength)
        window1.render_complete()

    glfw.terminate()

if __name__ == "__main__":
    main()
