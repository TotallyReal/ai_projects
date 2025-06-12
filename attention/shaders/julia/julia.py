import math
import random
import glfw
import numpy as np
from typing import Tuple, Optional
import os

from attention.shaders.shader import GlWindow, UniformType, IndexedSaver, ShaderInfo

Julia_shader_info = ShaderInfo.load_from(
    vert_path='julia.vert', frag_path='julia.frag', folder=os.path.dirname(os.path.abspath(__file__)),
    uniform_types=[
        ('u_dimension',  UniformType.VEC2),
        ('u_position',   UniformType.VEC2),
        ('u_constant',   UniformType.VEC2),
        ('u_color_wave', UniformType.FLOAT),
    ]
)

class JuliaWindow(GlWindow):
    
    def __init__(self, size: int):
        super().__init__(
            Julia_shader_info,
            width=size, height=size, title='Julia set')

        self.uniforms = {
            'u_position': [-1.2, -1.2],  # Bottom left corner
            'u_dimension': [2.4, 2.4],  # Width, height
            'u_constant': [-0.37, -0.36],
            'u_color_wave': 0
        }

        glfw.set_window_aspect_ratio(self.window, 1, 1)

        self._saver = IndexedSaver(os.path.join(current_dir, 'data'), 'julia')

        self.mouse_events.mouse_scrolled_listeners.append(self.scroll_callback)
        self.mouse_events.mouse_dragged_listeners.append(self.dragged_callback)

        # def window_size_callback(window, width, height):
        #     print(f'Window size: {width}, {height}')
        #
        # def framebuffer_size_callback(window, width, height):
        #     print(f'buffer size: {width}, {height}')
        #
        # glfw.set_window_size_callback(self.window, window_size_callback)
        # glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)


    def open_window(self):
        self.draw_screen(self.uniforms)
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            # time = glfw.get_time() - start_time

            # Key handling
            if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                image = self.generate_image()
                self._saver.save_image(image)

                glfw.wait_events_timeout(0.2)  # debounce

            # if glfw.get_key(self.window, glfw.KEY_ENTER) == glfw.PRESS:
            #     random_julia(fixed_constant=(-0.37, -0.36))
            #
            #     glfw.wait_events_timeout(0.2)  # debounce

            # if glfw.get_key(app_window.window, glfw.KEY_SPACE) == glfw.PRESS:
            #     generate_images_data(1000)
            #
            #     glfw.wait_events_timeout(0.2)  # debounce


        glfw.terminate()

    # <editor-fold desc="Control position of the Julia fractal">

    def zoom_to_point(self, x: float, y: float, zoom: float):
        t = 1 / zoom
        u_position = self.uniforms['u_position']
        self.uniforms['u_position'] = [u_position[0] * t + x * (1 - t), u_position[1] * t + y * (1 - t)]
        u_dimension = self.uniforms['u_dimension']
        self.uniforms['u_dimension'] = [u_dimension[0] * t, u_dimension[1] * t]
        self.draw_screen(self.uniforms)

    def move_drawing_window(self, dx: float, dy: float):
        self.uniforms['u_position'][0] += dx
        self.uniforms['u_position'][1] += dy
        self.draw_screen(self.uniforms)

    # </editor-fold>

    # <editor-fold desc="Mouse Events">

    def scroll_callback(self, window: GlWindow, xpos: float, ypos: float, scroll_value: float):
        u_position = self.uniforms['u_position']
        u_dimension = self.uniforms['u_dimension']

        width, height = glfw.get_window_size(self.window)
        x = linear_map(xpos, 0, width, u_position[0], u_position[0] + u_dimension[0])
        y = linear_map(ypos, 0, height, u_position[1], u_position[1] + u_dimension[1])
        self.zoom_to_point(x, y, math.exp(scroll_value / 2))

    def dragged_callback(self, window: GlWindow, xpos: float, ypos: float, dx: float, dy: float):
        u_dimension = self.uniforms['u_dimension']
        u_position = self.uniforms['u_position']

        width, height = glfw.get_window_size(self.window)
        dx *= u_dimension[0] / width
        dy *= u_dimension[1] / height

        self.uniforms['u_position'] = [u_position[0] - dx, u_position[1] - dy]

        self.draw_screen(self.uniforms)

    # </editor-fold>

    # <editor-fold desc="Random Julia Sets">

    def random_labeled_julia(
            self, seed: int = -1,
            fixed_constant: Optional[Tuple[float, float]] = None,
            zoom_steps: int = -1):
        self.random_julia(seed=seed, fixed_constant=fixed_constant, zoom_steps=zoom_steps)
        image = np.array(self.generate_image().convert("L"))
        return image, tuple(self.uniforms['u_constant']) + tuple(self.uniforms['u_position']) + (self.uniforms['u_dimension'][0],)

    def random_julia(
            self, seed: int = -1,
            fixed_constant: Optional[Tuple[float, float]] = None,
            zoom_steps: int = -1):
        """
        This is a description
        :return:
        """

        if seed < 0:
            seed = random.randint(0, 1000000000)
        random.seed(seed)

        if fixed_constant is None:
            self.uniforms['u_constant'] = [random.uniform(-0.5, -0.3), random.uniform(-0.5, 0.5)]
        else:
            self.uniforms['u_constant'] = list(fixed_constant)

        # uniforms['u_constant']  = [random.uniform(-0.5,-0.3), random.uniform(-0.5,0.5)]
        self.uniforms['u_position'] = [-1.2, -1.2]
        diameter = 2.4
        self.uniforms['u_dimension'] = [diameter, diameter]

        # random.seed(0)
        if zoom_steps < 0:
            zoom_steps = random.choice([0, 1, 2, 3, 4])
        # print(f'{zoom_steps=}')
        for step in range(zoom_steps + 1):
            self.draw_screen(self.uniforms)
            image = np.array(self.generate_image().convert("L"))
            row, col = choose_random_edge(image, 0.9)

            center = (
                self.uniforms['u_position'][0] + diameter * col / image.shape[1],
                self.uniforms['u_position'][1] + diameter * row / image.shape[0]
            )

            diameter /= 1.2
            self.uniforms['u_dimension'] = [diameter, diameter]

            self.uniforms['u_position'] = [center[0] - diameter / 2, center[1] - diameter / 2]
        self.draw_screen(self.uniforms)


    # </editor-fold>




# def generate_images_data(n: int):
#     images = []
#     labels = []
#     for image_index in range(n):
#         random_julia(zoom_steps=0, fixed_constant=(-0.37, -0.36))
#         image = np.array(app_window.generate_image().convert("L"))
#         images.append(image)
#         labels.append(
#             tuple(uniforms['u_constant']) + tuple(uniforms['u_position']) + (uniforms['u_dimension'][0],)
#         )
#         print(f'Saved image {image_index}')
#
#     images = np.array(images)
#     labels = np.array(labels)
#
#     data = {
#         'images': torch.from_numpy(images).unsqueeze(1).float(),
#         'labels': torch.from_numpy(labels).float()
#     }
#     torch.save(data, 'data/dataset_only_position.pt')

def linear_map(value: float, from_start: float, from_end: float, to_start: float, to_end: float):
    return to_start + (value-from_start) * (to_end-to_start) / (from_end-from_start)


