import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.signal import convolve2d
from vision import filters

def visual_derivative():
    """
    Plots a function, its first, and second derivatives.
    """

    T = 10

    def smooth_step(x):
        return 1 / (1 + np.exp(-T * x))

    def d_smooth_step(x):
        return T * np.exp(-T * x) / (1 + np.exp(-T * x)) ** 2

    def dd_smooth_step(x):
        return -T ** 2 * np.exp(T * x) * (np.exp(T * x) - 1) / (1 + np.exp(T * x)) ** 3

    x = np.linspace(-2, 2, 400)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    # Plot the step function
    axs[0].plot(x, smooth_step(x), color='b')
    axs[0].axvline(x=0, color='r', linestyle='--')
    axs[0].set_title('Edge from 0 to 1')

    # Plot the first derivative
    axs[1].plot(x, d_smooth_step(x), label="1st Derivative", color='orange')
    axs[1].axvline(x=0, color='r', linestyle='--')
    axs[1].set_title('First derivative')

    # Plot the second derivative
    axs[2].plot(x, dd_smooth_step(x), label="2nd Derivative", color='purple')
    axs[2].axvline(x=0, color='r', linestyle='--')
    axs[2].set_title('Second derivative')

    # Display the plots
    plt.tight_layout()
    plt.show()

def show_harris_corner_gradients(image: np.ndarray, gradient_filters: Tuple[np.ndarray, np.ndarray]):
    """
    Plots:
    1) An image with its gradients on every pixel.
    2) The distribution of the gradients in 2D, together with the main and second axis.
    """
    gradients = np.stack((
        convolve2d(image, gradient_filters[0], mode='same', boundary='fill', fillvalue=0),
        convolve2d(image, gradient_filters[1], mode='same', boundary='fill', fillvalue=0)
    ), axis=-1)

    # remove the boundaries and their noise
    image = image[3:-3, 3:-3]
    gradients = gradients[3:-3, 3:-3]

    # Normalize so that max gradient will be of length 1, so we have the right scale to draw it
    grad_mag_sq = np.sum(gradients ** 2, axis=2)
    normalized_gradients = gradients/np.sqrt(np.max(grad_mag_sq))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap='gray')
    h, w = image.shape
    axes[0].set_xlim([-0.5,w-0.5])
    axes[0].set_ylim([h-0.5,-0.5])
    for row_index, gradients_in_row in enumerate(normalized_gradients):
        for col_index, gradient in enumerate(gradients_in_row):
            axes[0].arrow(x=col_index, y=row_index, dx=2 * gradient[0], dy=2 * gradient[1], head_width=0.2,
                          edgecolor='red')

    gradients = gradients.reshape(-1, 2).T
    # When drawing images, the origin is the top left corner and the "positive" quadrant is on the bottom right.
    # In standard plotting, it is the top right, so flip the y-coordinate
    gradients *= np.array([[1], [-1]])
    plot_2d_sample(axes[1], gradients)

    plt.show()

def normal_sample_2d(sample_size: int, center: Tuple[float, float], ev: Tuple[float,float], angle: float) -> np.ndarray:
    """
    Generate sample_size points in the plane with the given parameters
    """
    samples = np.random.normal(size=(2, sample_size))
    samples = np.array([[ev[0]],[ev[1]]]) * samples
    samples = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) @ samples
    samples = np.array([[center[0]],[center[1]]]) + samples
    return samples

def plot_2d_sample(ax, samples:np.ndarray):
    """
    Given the sample points in 2D, plots them, and a "bounding" ellipse according to the main
    and second eigenvalue\vectors of the sample.
    """
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
    e_vectors *= np.sqrt(e_values)

    ax.axhline(0, color='black', linewidth=1)  # Draw x-axis
    ax.axvline(0, color='black', linewidth=1)  # Draw y-axis
    ax.grid(True)

    ax.scatter(x=samples[0], y=samples[1], s=1)
    bound = np.max(np.abs(samples))
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])

    angle = 180 * np.atan2(e_vectors[1, 0], e_vectors[0, 0]) / np.pi

    scale = 2
    ax.arrow(0, 0, *(scale*e_vectors[:, 0]), width=bound/80, color='black')
    ax.arrow(0, 0, *(scale*e_vectors[:, 1]), width=bound/80, color='black')
    ax.add_patch(patches.Ellipse(
        xy=(0, 0), width=2 * scale * np.sqrt(e_values[0]), height=2 * scale *np.sqrt(e_values[1]), angle=angle,
        facecolor='none', edgecolor='red'))

def plot_image_grid(image_grid: List[List[np.ndarray]], titles: List[str], image_scale: float = 2.5):
    n_cols = len(image_grid)
    n_rows = max([len(image_col) for image_col in image_grid])
    h, w = image_grid[0][0].shape[:2]

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(image_scale * n_cols * w / h, image_scale * n_rows))
    if n_cols * n_rows == 1:
        axes = np.array([axes])
    axes = axes.T

    for axes_col, title, image_col in zip(axes, titles, image_grid):
        if n_rows == 1:
            axes_col = [axes_col]
        axes_col[0].set_title(title)
        for ax, img in zip(axes_col, image_col):
            ax.imshow(img, cmap='gray')
        for ax in axes_col:
            ax.axis('off')

    fig.tight_layout()
    plt.show()

def show_edge_derivatives(gray_image: np.ndarray):
    titles = ['original']
    image_grid = [[gray_image]]
    for title, grad_filters in filters.edge_derivatives.items():
        titles.append(title)
        dx = convolve2d(gray_image, grad_filters[0], mode='same', boundary='fill', fillvalue=0)
        dy = convolve2d(gray_image, grad_filters[1], mode='same', boundary='fill', fillvalue=0)
        grad_size = np.sqrt(dx**2 + dy**2)
        image_grid.append([dx, dy, grad_size])

def show_edge_laplacian(gray_image: np.ndarray):
    titles = ['original']
    image_grid = [[gray_image]]
    for title, filter in filters.edge_laplacians.items():
        image_grid.append([np.abs(convolve2d(gray_image, filter, mode='same', boundary='fill', fillvalue=0))])
        titles.append(title)

    plot_image_grid(image_grid, titles, 3.5)

def show_blurs(gray_image: np.ndarray, dimensions:List[int]):
    titles = ['original']
    grid = [[gray_image]]
    for dimension in dimensions:
        blurred_image = convolve2d(gray_image, filters.binomial_blur2d(dimension), mode='same', boundary='fill', fillvalue=0)
        laplacian_image = np.abs(convolve2d(blurred_image, filters.edge_laplacians['4 directions'], mode='same', boundary='fill', fillvalue=0))
        grid.append([blurred_image, laplacian_image])
        titles.append(f'dim={dimension}')
    plot_image_grid(grid, titles)

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

