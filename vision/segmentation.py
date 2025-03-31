import numpy as np
from typing import Optional, Tuple
from scipy.spatial import cKDTree

def k_mean_segmentation(
        k: int, points: np.ndarray, initial_means: Optional[np.ndarray] = None, max_iterations: int = 10)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    k: number of segmenta
    points: n x d - n points in dimension d
    initial_means: k x d - k points, or None from randomly choose the initialization

    Returns
    indices : vector of length n with entries in [0,1,...,k-1]
    means : The final k-means in a k x d array
    """
    d = points.shape[1]

    if initial_means is None:
        max_values = np.max(points, axis=0)
        min_values = np.min(points, axis=0)
        means = min_values + (max_values - min_values) * np.random.rand(k, d)
    else:
        assert k, d == initial_means.shape
        means = initial_means

    indices = np.zeros((len(points),))

    for _ in range(max_iterations):
        tree = cKDTree(means)
        distances, indices = tree.query(points, k=1)

        sums = np.zeros((k,d))
        np.add.at(sums, indices, points)
        counters = np.bincount(indices, minlength=k)
        zero_counters = (counters == 0)
        counters[zero_counters] = 1
        new_means = sums/counters.reshape(k,1)
        new_means[zero_counters] = means[zero_counters]

        max_norm_change = np.max(np.linalg.norm(means - new_means, axis=1))
        means = new_means
        if max_norm_change < 1:
            break

    return indices, means


def segmentation_by_color_and_position(image: np.ndarray, k: int, pos_weight: float = 1) -> np.ndarray:

    positions = np.stack(np.indices(image.shape[:2]), -1).astype(float)
    image_with_positions = np.concatenate([image, pos_weight*positions], axis=-1)
    points = image_with_positions.reshape(-1,5)
    indices, means = k_mean_segmentation(k=k, points=points)
    return indices.reshape(image.shape[:2])

