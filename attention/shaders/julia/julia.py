import math
import random
import torch
import glfw
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from typing import Tuple

from shader import GlWindow, UniformType, IndexedSaver

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


def f(x: float, y: float):
    xx = 0.0
    yy = 0.0
    for k in range(64):
        if x*x+y*y > 16:
            return k, x*x+y*y
        xx = x*x-y*y-uniforms['u_constant'][0]
        yy = 2.0*x*y-uniforms['u_constant'][1]
        x = xx
        y = yy
    return -1, 1

class JuliaException (Exception):
    pass

def random_edge_position(n: int):
    x1, y1 = random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)
    for attempt in range(1000):
        if 0<=f(x1, y1)[0]<5:
            break
        x1, y1 = random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)
    else:
        raise JuliaException('Could not find a suitable starting position')

    x2, y2 = random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)
    for attempt in range(1000):
        value = f(x2, y2)[0]
        if value>60 or value == -1:
            break
        x2, y2 = random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)
    else:
        raise JuliaException('Could not find a suitable starting position')

    prev = f(x1, y1)[0]
    best_grad = 0
    best_pos = []
    for t in range(n + 1):
        x = x1 + (x2-x1) * t/n
        y = y1 + (y2-y1) * t/n
        k, _ = f(x,y)
        curr_grad = k - prev
        if abs(curr_grad) == best_grad:
            best_pos.append((x1 + (x2-x1) * (2*t-1)/(2*n), y1 + (y2-y1) * (2*t-1)/(2*n)))
        elif abs(curr_grad) > best_grad:
            best_pos = [(x1 + (x2 - x1) * (2 * t - 1) / (2 * n), y1 + (y2 - y1) * (2 * t - 1) / (2 * n))]

    return random.choice(best_pos)


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
    Given a grayscale image, returns a random pixel position where the value is above the p - percentile
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


def random_julia(zoom_steps: int = -1):
    """
    This is a description
    :return:
    """
    # uniforms['u_constant']  = [random.uniform(-0.5,-0.3), random.uniform(-0.5,0.5)]
    uniforms['u_position']  = [-1.2, -1.2]
    diameter = 2.4
    uniforms['u_dimension'] = [diameter, diameter]

    # random.seed(0)
    if zoom_steps < 0:
        zoom_steps = random.choice([0,1,2,3,4])
    # print(f'{zoom_steps=}')
    for step in range(zoom_steps + 1):

        draw_screen()
        image = np.array(app_window.generate_image().convert("L"))
        row, col = choose_random_edge(image, 0.9)

        # <editor-fold desc=" ------------------------ plot ------------------------">

        # image = np.stack((image,)*3, axis=-1)
        # h, w, _ = image.shape
        # half = 5
        # r_start = max(0, row - half)
        # r_end = min(h, row + half + 1)
        # c_start = max(0, col - half)
        # c_end = min(w, col + half + 1)
        #
        # image[r_start:r_end, c_start:c_end] = [255, 0, 0]
        # plt.imshow(image)
        # plt.show()
        # print(f'{row=} , {col=}')
        # </editor-fold>

        center = (
            uniforms['u_position'][0] + diameter * col / image.shape[1],
            uniforms['u_position'][1] + diameter * row / image.shape[0]
        )

        diameter /= 1.2
        uniforms['u_dimension'] = [diameter, diameter]

        uniforms['u_position'] = [center[0]-diameter/2, center[1]-diameter/2]
    draw_screen()

def generate_images_data(n: int):
    images = []
    labels = []
    for image_index in range(n):
        random_julia(0)
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
    draw_screen()

def moveDrawingWindow(dx: float, dy: float):
    uniforms['u_position'][0] += dx
    uniforms['u_position'][1] += dy
    draw_screen()

def linear_map(value: float, from_start: float, from_end: float, to_start: float, to_end: float):
    return to_start + (value-from_start) * (to_end-to_start) / (from_end-from_start)

def scroll_callback(window: GlWindow, xpos: float, ypos: float, scroll_value: float):
    u_position = uniforms['u_position']
    u_dimension = uniforms['u_dimension']
    x = linear_map(xpos, 0, app_window.width , u_position[0], u_position[0] + u_dimension[0])
    y = linear_map(ypos, 0, app_window.height, u_position[1], u_position[1] + u_dimension[1])
    zoomToPoint(x, y, math.exp(scroll_value/2))



def dragged_callback(window: GlWindow, xpos: float, ypos: float, dx: float, dy: float):
    u_dimension = uniforms['u_dimension']
    u_position = uniforms['u_position']

    dx *= u_dimension[0] / window.width
    dy *= u_dimension[1] / window.height

    uniforms['u_position'] = [u_position[0] - dx, u_position[1] - dy]

    draw_screen()

def draw_screen():
    app_window.render_start()

    for name, value in uniforms.items():
        app_window.set_uniform(name, value)

    app_window.render_complete()


app_window.mouse_events.mouse_scrolled_listeners.append(scroll_callback)
app_window.mouse_events.mouse_dragged_listeners.append(dragged_callback)

saver = IndexedSaver('data', 'julia')

start_time = glfw.get_time()

draw_screen()

while not glfw.window_should_close(app_window.window):
    glfw.poll_events()
    time = glfw.get_time() - start_time

    # Key handling
    if glfw.get_key(app_window.window, glfw.KEY_S) == glfw.PRESS:
        image = app_window.generate_image()
        saver.save_image(image)

        glfw.wait_events_timeout(0.2)  # debounce

    if glfw.get_key(app_window.window, glfw.KEY_ENTER) == glfw.PRESS:
        random_julia()

        glfw.wait_events_timeout(0.2)  # debounce

    if glfw.get_key(app_window.window, glfw.KEY_SPACE) == glfw.PRESS:
        generate_images_data(1000)

        glfw.wait_events_timeout(0.2)  # debounce

    # Render
    # draw_screen()

glfw.terminate()

