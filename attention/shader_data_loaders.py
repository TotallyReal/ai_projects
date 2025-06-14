from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

class PositionCropDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        positions: List[Tuple[int, int]],
        diameter: int,
        bounding_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        transform: Optional[callable] = None
    ):
        """
        :param image: np array
        :param positions: list of (x, y) pixel positions
        :param diameter: size of square crop (in pixels)
        :param bounding_box: ((xmin, xmax), (ymin, ymax)) for normalized labels, optional
        :param transform: optional image transform (e.g., ToTensor, Normalize)
        """
        if image.ndim != 2:
            raise ValueError("Image must be a NumPy array with shape (H, W).")
        self.image = image
        self.img_width, self.img_height = self.image.shape[:2]

        self.radius = diameter // 2

        self.padded_image = np.pad(
            self.image,
            ((self.radius, self.radius), (self.radius, self.radius)),
            mode='constant',
            constant_values=0
        )

        self.positions = positions
        self.diameter = diameter
        self.transform = transform

        if bounding_box is not None:
            self.min_x = bounding_box[0][0]
            self.rx = (bounding_box[0][1] - bounding_box[0][0])/self.img_width
            self.min_y = bounding_box[1][0]
            self.ry = (bounding_box[1][1] - bounding_box[1][0])/self.img_height
        else:
            self.min_x = 0
            self.rx = 1
            self.min_y = 0
            self.ry = 1

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.positions[idx]

        crop = self.padded_image[y : y + self.diameter, x : x + self.diameter]

        # image transform
        if self.transform:
            crop = self.transform(crop)

        # label transform
        label_x = self.min_x + x * self.rx
        label_y = self.min_y + y * self.ry
        label = torch.tensor([label_x, label_y], dtype=torch.float32)

        return crop, label

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

def edginess(image: np.ndarray) -> np.ndarray:
    dx = convolve2d(image, sobel[0], mode='same', boundary='symm', fillvalue=0)
    dy = convolve2d(image, sobel[1], mode='same', boundary='symm', fillvalue=0)
    norm_sq = dx * dx + dy * dy
    return np.sqrt(norm_sq)

def edge_positions(image: np.ndarray, radius: int, edginess: int = 500000) -> np.ndarray:
    """
    Given a grayscale image, computes the positions of edges, with 'edginess' value higher than p of the pixels.
    :param image: a 2D numpy array
    :param p:     float in [0,1]
    :return:      An array (n,2) of positions
    """
    dx = convolve2d(image, sobel[0], mode='same', boundary='symm', fillvalue=0)
    dy = convolve2d(image, sobel[1], mode='same', boundary='symm', fillvalue=0)
    norm_sq = dx * dx + dy * dy
    # import matplotlib.pyplot as plt
    # plt.imshow(norm_sq, cmap='gray')
    # plt.show()
    # plt.hist(norm_sq.reshape(-1), bins=100)
    # plt.show()
    # mask = (norm_sq >= np.quantile(norm_sq, p))
    mask = norm_sq > edginess
    mask = binary_dilation(mask, structure=disk(radius//2))
    mask[:radius, :]  = False
    mask[-radius:, :] = False
    mask[:, radius]   = False
    mask[:, -radius:] = False
    indices = np.argwhere(mask)
    return indices
    # seed = np.random.randint(0,10000000)
    # # seed = 0
    # # print(f'{seed=}')
    # np.random.seed(seed)
    # return tuple(indices[np.random.choice(len(indices))])