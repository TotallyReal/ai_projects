import math
import random
import torch
import glfw
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from attention.shaders.shader import GlWindow, UniformType, IndexedSaver

width = 256
height = 256

app_window = GlWindow(
    width, height,
    "Julia set", "julia.vert", "julia.frag",
    [
        ('u_resolution', UniformType.VEC2),
        ('u_dimension',  UniformType.VEC2),
        ('u_position',   UniformType.VEC2),
        ('u_constant',   UniformType.VEC2),
        ('u_density',    UniformType.FLOAT),
        ('u_color_wave', UniformType.FLOAT),
    ]
)

uniforms = {
    'u_resolution': [width, height],
    'u_position'  : [-1.2,-1.2],        # Bottom left corner
    'u_dimension' : [2.4,2.4],          # Width, height
    'u_constant'  : [-0.37, -0.36],
    'u_density'   : 1,
    'u_color_wave': 0
}






sobel = (
    np.array([
        [1, 2,  0, -2, -1],
        [2, 3,  0, -3, -2],
        [3, 5,  0, -5, -3],
        [2, 3,  0, -3, -2],
        [1, 2,  0, -2, -1]
    ]),
    np.array([
        [ 1,  2,  3,  2,  1],
        [ 2,  3,  5,  3,  2],
        [ 0,  0,  0,  0,  0],
        [-2, -3, -5, -3, -2],
        [-1, -2, -3, -2, -1]
    ]))

def choose_random_edge(image: np.ndarray, p: float) -> Tuple[int, int]:
    """
    Given a grayscale image, returns a random pixel position where 'edginess' value is above the p - percentile
    :param image: a 2D numpy array
    :param p:     float in [0,1]
    :return:      A position (row, column)
    """
    dx = convolve2d(image, sobel[0], mode='same', boundary='symm', fillvalue=0)
    dy = convolve2d(image, sobel[1], mode='same', boundary='symm', fillvalue=0)
    norm_sq = dx * dx + dy * dy
    mask = (norm_sq >= np.quantile(norm_sq, p))
    # plt.imshow(mask, cmap='grey')
    # plt.show()
    indices = np.argwhere(mask)
    seed = np.random.randint(0,10000000)
    # seed = 0
    # print(f'{seed=}')
    np.random.seed(seed)
    return tuple(indices[np.random.choice(len(indices))])

def random_labeled_julia(
        seed: int = -1,
        fixed_constant: Optional[Tuple[float, float]] = None,
        zoom_steps: int = -1):
    random_julia(seed=seed, fixed_constant=fixed_constant, zoom_steps=zoom_steps)
    image = np.array(app_window.generate_image().convert("L"))
    return image, tuple(uniforms['u_constant']) + tuple(uniforms['u_position']) + (uniforms['u_dimension'][0],)

def random_julia(
        seed: int = -1,
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
        uniforms['u_constant']  = [random.uniform(-0.5,-0.3), random.uniform(-0.5,0.5)]
    else:
        uniforms['u_constant'] = list(fixed_constant)

    # uniforms['u_constant']  = [random.uniform(-0.5,-0.3), random.uniform(-0.5,0.5)]
    uniforms['u_position']  = [-1.2, -1.2]
    diameter = 2.4
    uniforms['u_dimension'] = [diameter, diameter]

    # random.seed(0)
    if zoom_steps < 0:
        zoom_steps = random.choice([0,1,2,3,4])
    # print(f'{zoom_steps=}')
    for step in range(zoom_steps + 1):

        app_window.draw_screen(uniforms)
        image = np.array(app_window.generate_image().convert("L"))
        row, col = choose_random_edge(image, 0.9)

        center = (
            uniforms['u_position'][0] + diameter * col / image.shape[1],
            uniforms['u_position'][1] + diameter * row / image.shape[0]
        )

        diameter /= 1.2
        uniforms['u_dimension'] = [diameter, diameter]

        uniforms['u_position'] = [center[0]-diameter/2, center[1]-diameter/2]
    app_window.draw_screen(uniforms)

def generate_images_data(n: int):
    images = []
    labels = []
    for image_index in range(n):
        random_julia(zoom_steps=0, fixed_constant=(-0.37, -0.36))
        image = np.array(app_window.generate_image().convert("L"))
        images.append(image)
        labels.append(
            tuple(uniforms['u_constant']) + tuple(uniforms['u_position']) + (uniforms['u_dimension'][0],)
        )
        print(f'Saved image {image_index}')

    images = np.array(images)
    labels = np.array(labels)

    data = {
        'images': torch.from_numpy(images).unsqueeze(1).float(),
        'labels': torch.from_numpy(labels).float()
    }
    torch.save(data, 'data/dataset_only_position.pt')


def zoomToPoint(x: float, y: float, zoom: float):
    t = 1 / zoom
    u_position = uniforms['u_position']
    uniforms['u_position'] = [u_position[0] * t + x * (1 - t), u_position[1] * t + y * (1 - t)]
    u_dimension = uniforms['u_dimension']
    uniforms['u_dimension'] = [u_dimension[0] * t, u_dimension[1] * t]
    app_window.draw_screen(uniforms)

def moveDrawingWindow(dx: float, dy: float):
    uniforms['u_position'][0] += dx
    uniforms['u_position'][1] += dy
    app_window.draw_screen(uniforms)

def linear_map(value: float, from_start: float, from_end: float, to_start: float, to_end: float):
    return to_start + (value-from_start) * (to_end-to_start) / (from_end-from_start)

def scroll_callback(window: GlWindow, xpos: float, ypos: float, scroll_value: float):
    u_position = uniforms['u_position']
    u_dimension = uniforms['u_dimension']

    width, height = glfw.get_window_size(app_window.window)
    x = linear_map(xpos, 0, width , u_position[0], u_position[0] + u_dimension[0])
    y = linear_map(ypos, 0, height, u_position[1], u_position[1] + u_dimension[1])
    zoomToPoint(x, y, math.exp(scroll_value/2))



def dragged_callback(window: GlWindow, xpos: float, ypos: float, dx: float, dy: float):
    u_dimension = uniforms['u_dimension']
    u_position = uniforms['u_position']

    width, height = glfw.get_window_size(app_window.window)
    dx *= u_dimension[0] / width
    dy *= u_dimension[1] / height

    uniforms['u_position'] = [u_position[0] - dx, u_position[1] - dy]

    app_window.draw_screen(uniforms)

def window_size_callback(window, width, height):
    # uniforms['u_position'] = [width, height]
    # print(f"Updated resolution to {width}x{height}")
    print(f'Window size: {width}, {height}')
    pass

def framebuffer_size_callback(window, width, height):
    print(f'buffer size: {width}, {height}')


app_window.mouse_events.mouse_scrolled_listeners.append(scroll_callback)
app_window.mouse_events.mouse_dragged_listeners.append(dragged_callback)
glfw.set_window_size_callback(app_window.window, window_size_callback)
glfw.set_framebuffer_size_callback(app_window.window, framebuffer_size_callback)


saver = IndexedSaver('data', 'julia')

start_time = glfw.get_time()

glfw.set_window_aspect_ratio(app_window.window, 1, 1)
app_window.draw_screen(uniforms)

while not glfw.window_should_close(app_window.window):
    glfw.poll_events()
    time = glfw.get_time() - start_time

    # Key handling
    if glfw.get_key(app_window.window, glfw.KEY_S) == glfw.PRESS:
        image = app_window.generate_image()
        saver.save_image(image)

        glfw.wait_events_timeout(0.2)  # debounce

    if glfw.get_key(app_window.window, glfw.KEY_ENTER) == glfw.PRESS:
        random_julia(fixed_constant=(-0.37, -0.36))

        glfw.wait_events_timeout(0.2)  # debounce

    if glfw.get_key(app_window.window, glfw.KEY_SPACE) == glfw.PRESS:
        generate_images_data(1000)

        glfw.wait_events_timeout(0.2)  # debounce


glfw.terminate()

