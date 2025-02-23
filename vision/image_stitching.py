import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def draw_matches(image1: np.ndarray, image2: np.ndarray, threshold: float = 0.7):
    """
    Standard matching between two images using the open cv library.
    'threshold' should be a nonnegative number indicating how good of a matching we are looking for.
    threshold==0 means that we only allow perfect matching (which basically only possible if you match a picture
    with itself, and hope that no noise somehow got into the computations).
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches: List[cv2.DMatch] = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("SIFT Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Point = Tuple[float, float]  # for (x,y) points

def find_projective_transform(correspondence: List[Tuple[Point, Point]]) -> np.ndarray:
    """
    Given list of points (u,v), look for a projective map A which approximate the solution Au ~ v (as lines).
    """
    lines = []
    for (u1, u2), (v1, v2) in correspondence:
        # Under this map we have span(A (u1,u2,1)^T) = span((v1,v2,1))
        # This is equivalent to    (v_1,v_2,1) x A (u1,u2,1)^T = (0,0,0)
        lines.append(np.array([u1, u2, 1, 0, 0, 0, -u1 * v1, -u2 * v1, -v1]))
        lines.append(np.array([0, 0, 0, u1, u2, 1, -u1 * v2, -u2 * v2, -v2]))

    arr = np.array(lines)
    # If the rows for W are the lines generated above, and we consider A as a vector, then we are trying to solve:
    #           min |WA| , s.t. |A|=1
    # Since |WA|=|A^T W^T*W A|, then using the spectral decomposition of the positive semi-definite matrix W^TW,
    # we conclude that A is the eigenvector of W^T*W corresponding to the smallest eigenvalue.
    eigenvalues, eigenvectors = np.linalg.eigh(arr.T @ arr)
    return eigenvectors[:, 0].reshape(3, 3)

def point_matching(
        bf_matcher: cv2.BFMatcher,
        keypoints1: List[cv2.KeyPoint], descriptors1: List[np.ndarray],
        keypoints2: List[cv2.KeyPoint], descriptors2: List[np.ndarray],
        threshold: float = 0.5) -> List[Tuple[Point, Point]]:
    """
    Return matches (x,y) -> (u,v) from the first image to the second.
    """
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches: List[cv2.DMatch] = []
    for best_match, second_best in matches:
        if best_match.distance < threshold * second_best.distance:
            good_matches.append(best_match)

    # Create system of linear equations:
    return [(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in good_matches]

def proj_prod(matrix: np.ndarray, x, y):
    result = matrix @ np.array([x,y,1])
    return result[:2] / result[2]

class ImagePart:

    def __init__(self, image: np.ndarray, transform: np.ndarray):
        self.image = image
        self.transform = transform
        self.inv = np.linalg.inv(self.transform)
        self.h, self.w = image.shape[:2]
        self.is_rgb = len(image.shape) == 3

    def np_weights(self, world_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the world position, return two arrays indicating the 'weight' and 'image value' for each position.
        """
        # Apply the projective map to find the position in local space
        self_pos = np.tensordot(world_pos, self.inv.T, axes=1)
        self_pos = self_pos[:, :, :2] / self_pos[:, :, 2:3]

        self_pos = np.floor(self_pos).astype(int)       # TODO: change to some interpolation for better smoothening

        cols, rows = self_pos[:, :, 0], self_pos[:, :, 1]
        valid_mask = (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)      # stay inside the bounds

        shape = self_pos.shape[:2] + (3,) if self.is_rgb else self_pos.shape[:2]
        image_values = np.zeros(shape, dtype=self.image.dtype)
        image_values[valid_mask] = self.image[rows[valid_mask], cols[valid_mask]]

        # The larger the weight, the more inside the image we are
        weights = rows * (self.h - rows) * cols * (self.w - cols) * valid_mask.astype(int)
        if self.is_rgb:
            weights = weights[:,:,np.newaxis]

        return weights, image_values

    def transformed_corners(self):
        return [
            proj_prod(self.transform, 0, 0),
            proj_prod(self.transform, self.w, 0),
            proj_prod(self.transform, self.w, self.h),
            proj_prod(self.transform, 0, self.h)]


def stitch_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Stitch the images together. Assumes that all the images intersect in a meaningful way the
    first image (lots of interesting keypoints in the intersection).
    """
    is_rgb = len(images[0].shape)==3
    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images] if is_rgb else images

    # Generate SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    sift_data = [sift.detectAndCompute(img, None) for img in gray_images]
    bf = cv2.BFMatcher()

    # Find transforms from all images to the first image
    image_parts = [ImagePart(images[0], transform=np.identity(3))] + [
        ImagePart(image, transform=find_projective_transform(point_matching(bf, *sift_data[i+1], *sift_data[0])))
        for i, image in enumerate(images[1:])
    ]
    # TODO 1: Can improve how we do the match. For example, to avoid outliers, try to create a transform that fit really
    #         well most of the matches, rather one that tries to fit them all.
    #
    # TODO 2: Try to fit images not necessarily to the first one.

    corners = sum([image_part.transformed_corners() for image_part in image_parts], [])  # corners1 + corners2
    corners_x = [corner[0] for corner in corners]
    corners_y = [corner[1] for corner in corners]
    min_x = int(min(corners_x))
    max_x = int(max(corners_x))
    min_y = int(min(corners_y))
    max_y = int(max(corners_y))

    shape = (max_y - min_y + 1, max_x - min_x + 1)
    # Create the 3D array where  positions[i,j] = (min_x+i, min_y+j, 1)
    positions = np.dstack([np.indices(shape)[1], np.indices(shape)[0], np.ones(shape)]) + np.array([min_x, min_y, 0])

    if is_rgb:
        full_weights = np.zeros(shape=shape + (1,))
        full_image = np.zeros(shape=shape + (3,))
    else:
        full_weights = np.zeros(shape=shape)
        full_image = np.zeros(shape=shape)

    # Average over the different image parts
    for image_part in image_parts:
        part_weights, part_image_values = image_part.np_weights(positions)
        full_weights += part_weights
        full_image += part_image_values * part_weights

    full_weights[full_weights == 0] = 1
    full_image /= full_weights

    return full_image.astype(int)



# --------------------------------------- Example how to use: ---------------------------------------

images = [
    cv2.resize(
        cv2.imread(f'images/room/{i}.jpg', cv2.IMREAD_COLOR_RGB),
        dsize=(0,0), fx=0.1, fy=0.1)
    for i in [1,2,3,4]
]

if any(elem is None for elem in images):
    raise ValueError("Could not load one of the images.")

# To see the matches between pairs of images:
# draw_matches(images[0],images[1], threshold=0.5)

full_image = stitch_images(images)

plt.imshow(full_image, cmap='gray')

plt.show()