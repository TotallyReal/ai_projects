import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.signal import convolve2d

def normal_sample_2d(sample_size: int, center: Tuple[float, float], ev: Tuple[float,float], angle: float):
    samples = np.random.normal(size=(2, sample_size))
    samples = np.array([[ev[0]],[ev[1]]]) * samples
    samples = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) @ samples
    samples = np.array([[center[0]],[center[1]]]) + samples
    return samples

def plot_2d_sample(samples:np.ndarray):
    h, w = samples.shape[:2]
    assert h == 2

    mat = (samples @ samples.T) / w
    a = mat[0, 0]
    b = mat[1, 1]
    c = mat[0, 1]
    # 0 = (x-a)(x-b)-c*c = x^2 - (a+b)x + a*b-c*c
    # roots = (a+b)/2 +- sqrt((a-b)^2+4c^2)/2

    disc = (a - b) ** 2 + 4 * c ** 2
    e_values = np.array([(a + b + np.sqrt(disc)) / 2, (a + b - np.sqrt(disc)) / 2])
    e_vectors = np.array([
        [c, a - e_values[0]],
        [e_values[0] - a, c]
    ])
    d = np.sqrt(c * c + (e_values[0] - a) ** 2)
    e_vectors /= d
    e_vectors *= e_values

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axhline(0, color='black', linewidth=1)  # Draw x-axis
    plt.axvline(0, color='black', linewidth=1)  # Draw y-axis
    plt.grid(True)

    plt.scatter(x=samples[0], y=samples[1], s=1)
    bound = np.max(np.abs(samples))
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)

    angle = 180 * np.atan2(e_vectors[1, 0], e_vectors[0, 0]) / np.pi
    ax.add_patch(patches.Ellipse(
        xy=(0, 0), width=2 * e_values[0], height=2 * e_values[1], angle=angle,
        facecolor='none', edgecolor='red'))

    plt.arrow(0, 0, *e_vectors[:, 0], width=.08, color='black')
    plt.arrow(0, 0, *e_vectors[:, 1], width=.08, color='black')

    plt.show()

def corners(image: np.ndarray, window_size: int, threshold: float = 0.1):
    gray_image = image if len(image.shape)<3 else np.mean(image, axis=2)

    directions = [
        np.array([[1, -1]]),
        np.array([[1], [-1]]),
        np.array([[1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]

    window_ones = np.ones(shape=(window_size, window_size))

    edge_weight = []

    for direction in directions:
        arr = convolve2d(gray_image, direction, mode='same')
        arr = arr**2
        arr = convolve2d(arr, window_ones, mode='same')
        edge_weight.append(arr)

    # If a window contains at most one edge in on of the given directions above,
    # then along that edge the derivative is close to zero.

    min_arr = np.array(edge_weight).min(axis=0)
    max_value = np.max(min_arr)
    corners_arr = np.zeros(shape=gray_image.shape)
    corners_arr[min_arr>max_value * threshold] = 1
    y, x = np.where(min_arr>max_value * threshold)

    if len(image.shape) < 3:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.scatter(x, y, c="red", s=1)  # -y to match image orientation
    plt.show()
