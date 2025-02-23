from functools import cache
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple
from scipy.signal import convolve2d
from scipy.special import comb
from scipy.ndimage import convolve1d

# <editor-fold desc=" ------------------------ Blurs ------------------------">

@cache
def binomial_blur1d(dim: int):
    return np.array([comb(dim-1, k, exact=True) for k in range(dim)])/2**(dim-1)

@cache
def binomial_blur2d(dim: int):
    """
    A 2D symmetric binomial blur
    """
    binomials = binomial_blur1d(dim)[:,np.newaxis]
    return (binomials @ binomials.T)

@cache
def gaussian_blur1d(sigma: float) -> np.ndarray: # shape (2*ceil(3*sigma)+1,)
    # For a faster gaussian blur, you can use cv2.GaussianBlur
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()

# </editor-fold>

edge_derivatives = {
    'Roberts':(np.array([
        [0, 1],
        [-1,0]
               ]),np.array([
        [1,0],
        [0,-1]
        ])),
    'Prewitt':(np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
               ]),np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
        ])),
    'Sobel3':(np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
               ]),np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
        ])),
    'Sobel5':(np.array([
        [1, 2,  0, -2, -1],
        [2, 3,  0, -3, -2],
        [3, 5,  0, -5, -3],
        [2, 3,  0, -3, -2],
        [1, 2,  0, -2, -1]
            ]),np.array([
        [ 1,  2,  3,  2,  1],
        [ 2,  3,  5,  3,  2],
        [ 0,  0,  0,  0,  0],
        [-2, -3, -5, -3, -2],
        [-1, -2, -3, -2, -1]
        ])),
}

edge_laplacians = {
    '4 directions': np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
    ]),
    '8 directions': np.array([
        [1, 4, 1],
        [4,-20,4],
        [1, 4, 1]
    ]),
}

def harris_weight(
        gray_image: np.ndarray, gradient_filters:Tuple[np.ndarray, np.ndarray], window_size:int, k:float=0.05):
    h,w = gray_image.shape
    dx = convolve2d(gray_image, gradient_filters[0], mode='same', boundary='fill', fillvalue=0)
    dy = convolve2d(gray_image, gradient_filters[1], mode='same', boundary='fill', fillvalue=0)

    quadratics = np.empty((h,w,3))
    quadratics[...,0] = dx ** 2
    quadratics[...,1] = dx * dy
    quadratics[...,2] = dy ** 2

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
