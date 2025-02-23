# https://www.youtube.com/@firstprinciplesofcomputerv3258
from functools import cache
from scipy.special import comb

from scipy.signal import convolve2d
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from panorama import visualization

# Function to load all images from a directory
def load_images_from_directory(directory) -> List[np.ndarray]:
    images = []
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if not filename.startswith('low'):
            continue
        # Check if the file is an image (you can check for specific extensions like .jpg, .png, etc.)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Build the full file path
            img_path = os.path.join(directory, filename)
            # Read the image
            img = cv2.imread(img_path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Add the image to the list
                # w,h,_ = img.shape
                # resized_img = cv2.resize(img, (h//8, w//8))
                # output_path = os.path.join(directory, f'low_{filename}')
                # cv2.imwrite(output_path, resized_img)
    return images

# Example usage:
directory_path = "data"
images = load_images_from_directory(directory_path)

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
        scale_value = max(conv_image.max(), np.abs(conv_image.min()))
        conv_image *= (128/scale_value)
        edge_nomzlied = ((-1 <= conv_image) & (conv_image <= 1))
        conv_image -= conv_image.min()
        conv_image *= 255/conv_image.max()
        conv_image = conv_image.astype(np.uint8)

        axes[index + 1].imshow(edge_nomzlied, cmap='gray')
        axes[index + 1].axis('off')
    fig.tight_layout()
    plt.show()

img_path = 'data/castle.jpg'
image = cv2.imread(img_path)
if image is not None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.mean(image, axis=2)
# show_edge_derivatives(image)
# show_edge_laplacian(image)
# show_blurs(image, dimensions=[3,7,13])

image = convolve2d(image, blur(3), mode='same')
visualization.corners(image, 3,0.1)



# sample_size = 2000
# samples = visualization.normal_sample_2d(sample_size, center=(1,3), ev=(2,0.5), angle=2*np.pi*0.1)
# visualization.plot_2d_sample(samples)
