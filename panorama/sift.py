# https://www.youtube.com/@firstprinciplesofcomputerv3258
from functools import cache

from IPython.core.pylabtools import figsize
from scipy.special import comb

from scipy.signal import convolve2d
import os
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from panorama import visualization
import matplotlib.image as mpimg
from scipy.ndimage import maximum_filter
from numpy.lib.stride_tricks import sliding_window_view


# Function to load all images from a directory
def load_images_from_directory(directory) -> List[np.ndarray]:
    images = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image (you can check for specific extensions like .jpg, .png, etc.)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Build the full file path
            img_path = os.path.join(directory, filename)
            # Read the image
            img = mpimg.imread(img_path)
            if img is not None:
                images.append(img)  # Add the image to the list
                # w,h,_ = img.shape
                # resized_img = cv2.resize(img, (h//8, w//8))

                # output_path = os.path.join(directory, f'low_{filename}')
                # cv2.imwrite(output_path, resized_img)
    return images

# Example usage:
directory_path = "data"
# images = load_images_from_directory(directory_path)
# print(len(images))
# exit(0)

# Now images list contains all loaded images from the specified folder
# print(f"Loaded {len(images)} images.")

# fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12,2))
# axes: np.ndarray[plt.Axes] = axes.flatten()
#
# for ax, image in zip(axes, images):
#     ax.imshow(np.mean(image, axis=2), cmap='gray')
#     ax.axis('off')
 #
# plt.show()

@cache
def blur(dim: int):
    binomials = np.array([comb(dim-1, k, exact=True) for k in range(dim)]).reshape(dim, 1)

    return (binomials @ binomials.T)/ 4**(dim-1)

def show_blurs(gray_image: np.ndarray, dimensions:List[int]):
    h, w = gray_image.shape[:2]
    n_cols = 1 + len(dimensions)
    n_rows = 1
    img_scale = 3.5
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(img_scale * n_cols * w / h, img_scale * n_rows))

    axes[0].imshow(gray_image, cmap='gray')
    axes[0].axis('off')

    for index, dimension in enumerate(dimensions):
        blurred = convolve2d(gray_image, blur(dimension), mode='same', boundary='fill', fillvalue=0)
        blurred = convolve2d(blurred, edge_laplacians[0], mode='same', boundary='fill', fillvalue=0)

        axes[index + 1].imshow(blurred, cmap='gray')
        axes[index + 1].axis('off')
        axes[index + 1].set_title(f'dim={dimension}')
    fig.tight_layout()
    plt.show()

edge_derivatives = {
    'Roberts':(np.array([
        [0, 1],
        [-1,0]
               ]),np.array([
        [1,0],
        [0,-1]
        ])),
    'Prewitt':(np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
               ]),np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
        ])),
    'Sobel3':(np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
               ]),np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
        ])),
    'Sobel5':(np.array([
        [-1, -2,  0, 2, 2],
        [-2, -3,  0, 3, 2],
        [-3, -5,  0, 5, 3],
        [-2, -3,  0, 3, 2],
        [-1, -2,  0, 2, 1]
            ]),np.array([
        [ 1,  2,  3,  2,  1],
        [ 2,  3,  5,  3,  2],
        [ 0,  0,  0,  0,  0],
        [-2, -3, -5, -3, -2],
        [-1, -2, -3, -2, -1]
        ])),
}

def show_edge_derivatives(gray_image: np.ndarray):
    h, w = gray_image.shape[:2]
    n_cols = 1 + len(edge_derivatives)
    n_rows = 3
    img_scale = 3.5
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(img_scale * n_cols * w / h, img_scale * n_rows))

    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for index, filter_name in enumerate(edge_derivatives):
        conv_image_x = convolve2d(gray_image, edge_derivatives[filter_name][0], mode='same', boundary='fill', fillvalue=0)
        axes[0, index + 1].imshow(conv_image_x, cmap='gray')
        axes[0, index + 1].axis('off')
        axes[0, index + 1].set_title(filter_name)
        conv_image_y = convolve2d(gray_image, edge_derivatives[filter_name][1], mode='same', boundary='fill', fillvalue=0)
        axes[1, index + 1].imshow(conv_image_y, cmap='gray')
        axes[1, index + 1].axis('off')
        grad_size = np.sqrt(conv_image_x**2 + conv_image_y**2)
        axes[2, index + 1].imshow(grad_size, cmap='gray')
        axes[2, index + 1].axis('off')
    fig.tight_layout()
    plt.show()

edge_laplacians = [
    np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ]),
    np.array([
        [1, 4, 1],
        [4,-20,4],
        [1, 4, 1]
    ]),
]

def show_edge_laplacian(gray_image: np.ndarray):
    h, w = gray_image.shape[:2]
    n_cols = 1 + len(edge_laplacians)
    n_rows = 1
    img_scale = 3.5
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(img_scale * n_cols * w / h, img_scale * n_rows))

    axes[0].imshow(gray_image, cmap='gray')
    axes[0].axis('off')

    for index, filter in enumerate(edge_laplacians):
        conv_image = convolve2d(gray_image, filter, mode='same', boundary='fill', fillvalue=0)
        axes[index + 1].imshow(np.abs(conv_image), cmap='gray')
        axes[index + 1].axis('off')
        axes[index + 1].set_title(f'dim={len(filter)}')
    fig.tight_layout()
    plt.show()

img_path = 'images/knight.jpg'
image = mpimg.imread(img_path)
image = np.mean(image, axis=2)
# show_edge_derivatives(image)
# show_edge_laplacian(image)
# show_blurs(image, dimensions=[3,7,13])

# image = convolve2d(image, blur(3), mode='same')
# visualization.corners(image, 3,0.1)



# sample_size = 2000
# samples = visualization.normal_sample_2d(sample_size, center=(1,3), ev=(2,0.5), angle=2*np.pi*0.1)
# visualization.plot_2d_sample(samples)


def compute_gradients(
        gray_image: np.ndarray, gradient_filters:Tuple[np.ndarray, np.ndarray], window_size:int, k:float=0.05):
    h,w = gray_image.shape
    gradients = np.stack((
        convolve2d(gray_image, gradient_filters[0], mode='same', boundary='fill', fillvalue=0),
        convolve2d(gray_image, gradient_filters[1], mode='same', boundary='fill', fillvalue=0)
    ), axis=-1)

    quadratics = np.empty((h,w,3))
    quadratics[...,0] = gradients[...,0] ** 2
    quadratics[...,1] = gradients[...,0] * gradients[...,1]
    quadratics[...,2] = gradients[...,1] ** 2

    from numpy.lib.stride_tricks import sliding_window_view

    window_sums = sliding_window_view(quadratics, (window_size, window_size, 1))
    window_sums = window_sums.sum(axis=(3, 4))
    window_sums = window_sums[:, :, :, 0]

    a, b, c = window_sums[..., 0], window_sums[..., 1], window_sums[..., 2]
    trace = a + c
    determinant = a * c - b ** 2

    result = determinant - k * trace**2  # Shape (n-4, m-4)
    result -= np.min(result)
    result /= np.max(result)
    pad = window_size//2
    padded_result = np.pad(result, pad_width=((pad, window_size-pad), (pad, window_size-pad)), mode='constant', constant_values=0)

    return padded_result

def local_maximum(arr: np.ndarray, window_size: int):
    max_filtered = maximum_filter(arr, size=5, mode='constant', cval=-np.inf)
    return (arr == max_filtered)


# image = np.arange(48).reshape(8,6)
#
# image = np.zeros((30,30))
# image[10:20,10:20] = 1

def rectangle_contour(width: int, height: int, jump: int) -> np.ndarray:
    left = [(0, y) for y in range(0, height, jump)]
    top = [(x, height-1) for x in range(0, width, jump)]
    if top[0] == left[-1]:
        top.pop(0)

    right = [(width-1, height-1-y) for y in range(0, height, jump)]
    if right[0] == top[-1]:
        right.pop(0)

    bottom = [(width-1-x, 0) for x in range(0, width, jump)]
    if bottom[0] == right[-1]:
        bottom.pop(0)

    if left[0] == bottom[-1]:
        left.pop(0)
    return np.array(left+top+right+bottom)

def contour(arr: np.ndarray, initial_contour: np.ndarray, radius: int, weight1: float = 0.5, weight2: float = 0.5):
    h, w = arr.shape
    arr -= np.min(arr)
    arr /= np.max(arr)
    padded_arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=-np.inf)

    points = initial_contour
    points += radius * np.ones(shape=points.shape, dtype=int)

    n_points = len(points)
    print(n_points)
    points_by_step = [points]

    for step in range(60):
        new_points = []
        for index in range(n_points):
            prev_p = points[index-1]
            curr_p = points[index]
            next_p = points[index+1 if index+1<n_points else 0]

            best_new_position = curr_p
            arr_value = padded_arr[curr_p[1], curr_p[0]]
            deriv_1 = weight1*(np.sum((curr_p-prev_p) ** 2) + np.sum((curr_p-next_p) ** 2))
            deriv_2 = weight2*np.sum((next_p + prev_p-2*curr_p) ** 2)
            best_value = arr_value - deriv_1 - deriv_2
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    new_p = curr_p + np.array([i,j])
                    arr_value = padded_arr[new_p[1],new_p[0]]
                    deriv_1 = weight1*(np.sum((new_p-prev_p) ** 2) + np.sum((new_p-next_p) ** 2))
                    deriv_2 = weight2*np.sum((next_p + prev_p-2*new_p) ** 2)
                    value = arr_value - deriv_1 - deriv_2
                    if value > best_value:
                        best_value = value
                        best_new_position = new_p
            new_points.append(best_new_position)
        print(f'finished step {step}')

        points = np.array(new_points)
        points_by_step.append(points - radius * np.ones(shape=points.shape))

    # Set up figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(h-0.5, -0.5)
    ax.imshow(arr, cmap='gray')
    line, = ax.plot([], [], 'o-', lw=2)  # 'o-' plots both points and lines

    # Initialization function
    def init():
        x, y = points_by_step[0][0], points_by_step[0][1]
        line.set_data(x, y)
        return line,

    # Animation function (updates each frame)
    def update(frame):
        x, y = frame[:, 0], frame[:, 1]  # Extract x and y coordinates
        line.set_data(x, y)
        return line,

    import matplotlib.animation as animation
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=points_by_step, init_func=init, blit=True, interval=100)

    plt.show()
    # plt.imshow(arr, cmap='gray')
    # plt.xlim(-0.5, w-0.5)
    # plt.ylim(h-0.5, -0.5)
    # plt.scatter(x=points_by_step[-2][:,0], y=points_by_step[-2][:,1], color='red')
    # plt.scatter(x=points_by_step[-1][:,0], y=points_by_step[-1][:,1], color='blue')
    # plt.show()

# n = 19
# image = np.zeros(shape=(n,n))
# image[1:15, 6:13] = 1

# h,w = image.shape
# decrease_scale = 2
# image = image[0:h-(h%decrease_scale), 0:w-(w%decrease_scale)]
# h,w = image.shape
# image = image.reshape(h//decrease_scale, decrease_scale, w//decrease_scale, decrease_scale)
# image = np.mean(image, axis=3)
# image = np.mean(image, axis=1)
#
# image_x = convolve2d(image, edge_derivatives['Sobel3'][0], mode='valid', boundary='fill', fillvalue=0)
# image_y = convolve2d(image, edge_derivatives['Sobel3'][1], mode='valid', boundary='fill', fillvalue=0)
#
# grad_sq = image_x * image_x + image_y * image_y
# print(grad_sq.shape)
#
# h,w = grad_sq.shape
#
# jump = max(2 * 5, 2)
# jump = 4
# initial_contour=rectangle_contour(width=w, height=h, jump=8)
# contour(grad_sq, initial_contour=initial_contour, radius=5, weight1 = 0.05/jump**2, weight2 = 0.05/jump**2)

