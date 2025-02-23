import matplotlib.image as mpimg
import numpy as np
import os
from typing import List


def load_images_from_directory(directory: str) -> List[np.ndarray]:
    images = []
    for filename in os.listdir(directory):

        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)

            img = mpimg.imread(img_path)
            if img is not None:
                images.append(img)
    return images