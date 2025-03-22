import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from vision.image_stitching import Rect, apply_proj, change_domain
from vision.sift_compare import ImageSystem
from vision.misc import cached

import cProfile as profile
import logging
import glob

logger = logging.getLogger()

pr = profile.Profile()
pr.disable()


# TODO: clean this file ....

MIN_VALUE_GAP = 5
MIN_VALUE_DIAMETER = 15

# <editor-fold desc=" ------------------------ Basic algbera ------------------------">

def translation_mat(dx, dy) -> np.ndarray:
    return np.array([
        [1,0,dx],
        [0,1,dy],
        [0,0,1 ]
    ])

def rotation_mat(angle) -> np.ndarray:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [ cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [   0  ,   0  , 1]
    ])

def x_shear(y, z) -> np.ndarray:
    return np.array([
        [1,y,z],
        [0,1,0],
        [0,0,1]
    ])

def y_shear(x, z) -> np.ndarray:
    return np.array([
        [1,0,0],
        [x,1,z],
        [0,0,1]
    ])

def z_shear(x, y) -> np.ndarray:
    return np.array([
        [1,0,0],
        [0,1,0],
        [x,y,1]
    ])

def proj_prod(matrix: np.ndarray, x, y):
    result = matrix @ np.array([x,y,1])
    return result[:2] / result[2]

def proj_prod_vectors(matrix: np.ndarray, vectors: np.ndarray):
    result = matrix @ vectors
    return result[:2] / result[2]

# </editor-fold>

Point = Tuple[float, float]  # for (x,y) points


# <editor-fold desc=" ------------------------ Fundamental matrix ------------------------">


def generate_fundamental(matching: np.ndarray) -> np.ndarray:
    """
    Input: a matching array n x 2 x 2, where matching[i,1] -> matching[i,2] is the pixel matching between the pictures.

    Returns the 'best' approximation matrix F for the property:
            (x_1, y_1, 1) F (x_2, y_2, 1)^T = 0,
    where (x_1, y_1) -> (x_2, y_2) runs over all matching. F is returned normalized in the Frobenius norm.
    """
    # The linear system equivalent to the property above
    arr = np.array([[u1 * v1, u1 * v2, u1, u2 * v1, u2 * v2, u2, v1, v2, 1]
                    for (u1, u2), (v1, v2) in matching])

    # If the rows for W are the lines generated above, and we consider F as a vector, then we are trying to solve:
    #           min |WF| , s.t. |F|=1
    # Since |WF|=tr(F^T W^T*W F), using the spectral decomposition of the positive semi-definite matrix W^TW
    # we conclude that F is the eigenvector of W^T*W corresponding to the smallest eigenvalue.
    # This is equivalent to using the SVD method.
    eigenvalues, eigenvectors = np.linalg.eigh(arr.T @ arr)

    F = eigenvectors[:, 0].reshape(3, 3)

    # F should have rank 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt

    F /= np.linalg.norm(F)
    return F


def epipolar_distances(matching: np.ndarray, mat: np.ndarray) -> float:
    """
    For each matching (u,v) in R^2 x R^2, compute the distance of v from the epipolar line defined by (u,1) * mat.
    """
    matching1 = np.concatenate([matching, np.ones(shape=(len(matching), 2, 1))], axis=2)
    u, v = matching1[:, 0, :], matching1[:, 1, :]

    # Recall that the distance of a point (x0, y0) from a line ax+by+c=0 is
    #           |a*x0 + b*y0 + c| / sqrt(a^2+b^2)
    lines = u @ mat
    return np.abs(np.sum(lines * v, axis=1))/np.linalg.norm(lines[:, :2], axis=1)


def least_median_fundamental(matchings: np.ndarray) -> np.ndarray:
    """
    Search randomly for a fundamental matrix for the matching with small median, and return it.
    Input and output like in `generate_fundamental`.
    """
    sample_size = len(matchings)
    best_median = float('inf')
    best_fundamental = None
    attempts = 100
    while attempts > 0:
        attempts -= 1
        indices = np.random.choice(sample_size, 8, replace=False)
        matchings_sample = matchings[indices]
        fundamental_mat = generate_fundamental(matchings_sample)
        distances = epipolar_distances(matchings, fundamental_mat)
        median = np.median(distances)
        if best_median > median:
            best_median = median
            best_fundamental = fundamental_mat
            print(f'Improved median to {best_median}.')
            if best_median < 2:
                matchings = matchings[distances <= 4]
                sample_size = len(matchings)

    logger.info(f'Best median is {best_median}')  # : {np.array(sorted(indices))}')
    return best_fundamental


def epipole_to_infinity(F: np.ndarray, w: int, h: int):
    """
    Looks for a `good` matrix H such that FH has zero in the first column.
    Return H, FH
    """


    # correspondence between the (x1, y1, 1) to (x2, y2, 1) is by:
    #
    #       (x1, y1, 1) F (x2, y2, 1)^T = 0
    #
    # Applying g to the second image projectively, means that position
    #       (x2, y2, 1)^T -> (x3, y3, 1)^T ~ g(x2, y2, 1)^T (projectively)
    # Hence
    #       (x1, y1, 1) F g^-1 (x3, y3, 1)^T

    # We want the kernel to be mapped to infinity (1,0,0) so that the epipolar lines will be horizontal.
    # equivalently


    # ======================= Translation of center of image to the origin

    translation = translation_mat(-w/2, -h/2)

    translation_inv = translation_mat(w/2, h/2)

    F = F @ translation_inv

    # ======================= Shear kernel (epipole) to infinity

    U, S, Vt = np.linalg.svd(F)
    v = Vt.T[:,2]
    if v[2] == 0:
        ker_x, ker_y = v[:2]    # kernel already at infinity
    else:
        ker_x, ker_y = v[:2] / v[2:3]
        if abs(ker_x)/w < abs(ker_y)/h:
            if abs(ker_y)/h < 0.5:
                # TODO: If the epipole is not "too much" inside the picture, cut it down to a smaller picture,
                #       and run the algorithm on that
                raise Exception(f'Epipole should be outside the picture.')
            shear = z_shear(0, -1/ker_y)
            shear_inv = z_shear(0, 1/ker_y)
        else:
            if abs(ker_x)/w < 0.5:
                raise Exception(f'Epipole should be outside the picture.')
            shear = z_shear(-1/ker_x, 0)
            shear_inv = z_shear(1/ker_x, 0)

        F = F @ shear_inv

    # ======================= Rotate kernel (epipole) to be on the x-axis

    theta = np.atan2(ker_y, ker_x)
    if ker_x < 0:
        theta += np.pi

    rotation = rotation_mat(theta)

    F = F @ rotation.T
    F[:, 0] = 0

    H = rotation @ shear @ translation


    # theta = np.atan2(kernel[1], kernel[0])
    # if kernel[0] < 0:
    #     theta += np.pi
    #
    # rotation = rotation_mat(theta)
    # F = F @ rotation.T
    #
    # # ======================= Shear kernel to also have x = 0
    #
    # epipole_norm = np.linalg.norm(kernel)
    # sign = 1 if kernel[0] < 0 else -1
    #
    # shear = np.array([
    #     [1,0,0],
    #     [0,1,0],
    #     [sign/epipole_norm,0,1]
    # ])
    #
    # shear_inv = np.array([
    #     [1,0,0],
    #     [0,1,0],
    #     [-sign/epipole_norm,0,1]
    # ])
    #
    # F = F @ shear_inv
    # F[:, 0] = 0
    #
    # H = shear @ rotation @ translation

    return H, F


def y_align_images(domain1: Rect, domain2: Rect, fundamental: np.ndarray):
    """
    Find and return two `good` transformations H1, H2 such that
                        0  0  0
    H1^-T * F H2^-1 =   0  0  1
                        0 -1  0
    """
    h2, w2 = domain2.height, domain2.width
    H2, fundamental = epipole_to_infinity(fundamental, w2, h2)

    # Make the columns of the fundemantal matrix orthogonal
    v1, v2 = fundamental[:,1], fundamental[:,2]
    alpha = np.dot(v2, v1)/ np.dot(v1, v1)
    to_ortho = translation_mat(0, alpha)
    to_ortho_inv = translation_mat(0, -alpha)

    H2 = to_ortho @ H2
    fundamental = fundamental @ to_ortho_inv

    # Generate the (almost) orthogonal inverse

    v1, v2 = fundamental[:, 1], fundamental[:, 2]
    v0 = np.cross(v1, v2)

    H1 = np.vstack([v0, v2, -v1])
    corners = domain1.corners(projective=True)
    corners = H1 @ corners
    corners = (corners[:2] / corners[2:3]).T
    # This is a convex quadrilateral. To check if it kept the orientation, only need to check if all (or one) angle
    # is < pi.
    edge1 = corners[1] - corners[0]
    edge1_perp = np.array([-edge1[1], edge1[0]])
    edge2 = corners[2] - corners[1]
    if np.dot(edge1_perp, edge2) < 0:
        H1 = np.vstack([-v0, v2, -v1])

    return H1, H2


# </editor-fold>


# <editor-fold desc=" ------------------------ Matrix decomposition ------------------------">

def load_images_from(img_dir: str, img_type:str = 'jpg', resize_ratio: float = 1):
    return [
        cv2.resize(cv2.imread(file_name), dsize=(0,0), fx=resize_ratio, fy=resize_ratio)
        for file_name in glob.glob(f'{img_dir}/*.{img_type}')  # Update the path
    ]

def calibrate_from_images(images: List[np.ndarray], print_matrix: bool = True, show_chess_matches: bool = False):
    # Define the chessboard size (number of inner corners)
    chessboard_size = (9, 6)  # Adjust this based on your checkerboard pattern

    # Prepare object points (3D points in real-world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Lists to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    index = 0
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if show_chess_matches:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv2.imshow('Chessboard', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                        None)

    # Print the results
    if print_matrix:
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)

    return camera_matrix

"""
Input: list of points (u,v) correspondence between two images, which use the same camera calibration C.
Fix such (u,v) pair and let:
    1. w: the position in 3D relative to the first camera,
    2. P: the position in 3D of the second camera relative to the first
    3. K: the orientation of the second camera relative to the first (as orthogonal matrix).
          Formally: its columns are the left \ up \ forward directions
then:
    u ~ Cw
    v ~ CK^T(w-P)
therefore:
    C^{-1}u ~ w ~ P + KC^{-1}v

To find K and P, we first take the cross product with P:
    P x C^{-1}u ~ P x KC^{-1}v
Then inner product with C^{-1}u
    0 = <C^{-1}u, P x KC^{-1}v> = u^T C^{-T} P x KC^{-1}v

Since w -> P x w is linear, we can think of Px as a matrix, so set
F = C^{-T} P x KC^{-1}
E = Px K = C^T F C

Finally, we return P and K.
"""

def skew_ortho_decomposition(matrix: np.ndarray):
    """
    Try to approximate the given matrix as a product of (nonzero) skew-symmetric and orthogonal matrix.
    - The skew symmetric part is determined up to a sign,
    - The orthogonal part is determined up to a reflection.

    Returns (skew1, skew2), (ortho1, ortho2), skew x ortho
    """
    U, S, Vh = np.linalg.svd(matrix)
    # Removing noise: S should have two repeating entries and zero, and everything is up to scalar multiplication:
    s_value = (S[0] + S[1]) /2
    S = np.diag([s_value, s_value, 0])
    matrix = U @ S @ Vh

    sign = 1 if np.linalg.det(U) * np.linalg.det(Vh) > 0 else -1

    W = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, sign]
    ])

    K1 = U @ W @ Vh
    K2 = U @ W.T @ Vh
    skew = matrix @ K1.T  # = U @ (S @ W.T) @ U.T

    return (skew, -skew), (K1, K2), matrix

# </editor-fold>



def draw_corrsepondence(img1: np.ndarray, img2: np.ndarray, correspondence = {}, jump: int = 5):

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    original = axes[0].plot(0, 0, 'rx', markersize=10)[0]
    point = axes[1].plot(0, 0, 'rx', markersize=10)[0]

    def on_mouse_move(event):
        if event.inaxes == axes[0]:
            x, y = event.xdata, event.ydata
            x = int(x/jump) * jump
            y = int(y/jump) * jump
            if (x,y) in correspondence:
                original.set_data([x],[y])
                # print(f'{correspondence[(x,y)]=}')
                xx, yy = correspondence[(x,y)]
                point.set_data([xx], [yy])
            fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    plt.show()



# <editor-fold desc=" ------------------------ Image scans ------------------------">


class ImageMatchLookup:

    def __init__(self, image, radius: int, corners):
        self.windows = np.lib.stride_tricks.sliding_window_view(image, (2*radius+1, 2*radius+1), axis=(0,1))
        if len(image.shape) == 3:
            self.windows = self.windows.transpose((0, 1, 3, 4, 2))
        self.flat_windows = self.windows.reshape(*self.windows.shape[:2], -1)
        self.radius = radius
        self.vec_dim = 3 * (2*radius+1)**2   # TODO: what about gray images. Check where else in the code I need to update

        self.edges = list(zip(corners.T[:-1], corners.T[1:]))

    def get_match(self, y: int, x1: int, x2: int, patch):
        flat_sliding_windows = self.flat_windows[y - self.radius, x1:x2]
        # TODO: most of the time is spent in the next two lines. Try to find a faster way to do it
        flat_arr = flat_sliding_windows - patch
        flat_norm_sqr = np.einsum('ij,ij->i', flat_arr, flat_arr) / self.vec_dim  # faster than sum(arr**2,..)
        norm = np.sqrt(flat_norm_sqr)

        # Should be in [0, 255**2]
        min_index = np.argmin(norm)
        min_value = norm[min_index]

        if min_value > 35:
            return None
        indices = np.where(norm < min_value + MIN_VALUE_GAP)[0]
        if indices[-1] - indices[0] > MIN_VALUE_DIAMETER:
            return None

        return (x1 + self.radius + min_index, y)

    def in_bound(self, y):
        # Find intersection of line at height y with the second image
        xx2 = []
        for (e_x1, e_y1), (e_x2, e_y2) in self.edges:
            if e_y1 < y < e_y2 or e_y1 > y > e_y2:
                if e_x1 == e_x2:
                    xx2.append(e_x1)
                else:
                    # (y-e_y1)/(e_y2-e_y1) = (x-e_x1)/(e_x2-e_x1)
                    xx2.append(e_x1 + (e_x2 - e_x1) * (y - e_y1) / (e_y2 - e_y1))

        if len(xx2) != 2:
            return []
        xx2 = sorted(xx2)
        return [math.ceil(xx2[0]), math.floor(xx2[1])]
# def get_match(y: int, x1: int, x2: int, patch, radius, flat_windows):
#     flat_sliding_windows = flat_windows[y-radius, x1:x2]
#     # TODO: most of the time is spent in the next two lines. Try to find a faster way to do it
#     flat_arr = flat_sliding_windows-patch
#     flat_norm_sqr = np.einsum('ij,ij->i', flat_arr, flat_arr)/vec_dim  # faster than sum(arr**2,..)
#     norm = np.sqrt(flat_norm_sqr)
#
#     # Should be in [0, 255**2]
#     min_index = np.argmin(norm)
#     min_value = norm[min_index]
#
#     if min_value > 35:
#         return None
#     indices = np.where(norm < min_value + MIN_VALUE_GAP)[0]
#     if indices[-1] - indices[0] > MIN_VALUE_DIAMETER:
#         return None
#
#     return (x1+radius+min_index, y)
#
#
# def look_for_match(from_image, x, y, edges, radius):
#
#     # Find intersection of line at height y with the second image
#     xx2 = []
#     for (e_x1, e_y1), (e_x2, e_y2) in edges:
#         if e_y1 < y < e_y2 or e_y1 > y > e_y2:
#             if e_x1 == e_x2:
#                 xx2.append(e_x1)
#             else:
#                 # (y-e_y1)/(e_y2-e_y1) = (x-e_x1)/(e_x2-e_x1)
#                 xx2.append(e_x1 + (e_x2 - e_x1) * (y - e_y1) / (e_y2 - e_y1))
#
#     if len(xx2) != 2:
#         return None
#     xx2 = sorted(xx2)
#     xx2 = [math.ceil(xx2[0]), math.floor(xx2[1])]
#
#     if xx2[1]-2*radius <= xx2[0]:
#         return None
#
#     patch = from_image[y - radius:y + radius + 1, x - radius:x + radius + 1]
#     # GOD DAMMIT! You know how much time I wasted until I found out this was an unsigned integer?!
#     patch = patch.astype(np.int32).reshape(-1)
#
#     match = get_match(y, xx2[0], xx2[1] - 2 * radius, patch)
#     if match is not None:
#         return match
#     match = get_match(y+1, xx2[0], xx2[1] - 2 * radius, patch)
#     if match is not None:
#         return match
#     match = get_match(y-1, xx2[0], xx2[1] - 2 * radius, patch)
#     if match is not None:
#         return match
#     return None


def single_line_scan(x, y, from_image: np.ndarray, image_lookup, radius: int = 5, from_x=0, to_x=-1, y_error_bound: int = 0):
        # Find intersection of line at height y with the second image
        xx2 = image_lookup.in_bound(y)

        if len(xx2) != 2:
            return None

        if from_x < to_x:
            from_x = max(from_x, xx2[0])
            to_x = min(to_x, xx2[1])
        else:
            from_x = xx2[0]
            to_x = xx2[1]

        if to_x-2*radius <= from_x:
            return None

        patch = from_image[y - radius:y + radius + 1, x - radius:x + radius + 1].reshape(-1)
        # GOD DAMMIT! You know how much time I wasted until I found out this was an unsigned integer?!
        patch = patch.astype(np.int32)

        match = image_lookup.get_match(y, from_x, to_x - 2 * radius, patch)
        if match is not None:
            return match
        for dy in range(1, y_error_bound+1):
            match = image_lookup.get_match(y+dy, from_x, to_x - 2 * radius, patch)
            if match is not None:
                return match
            match = image_lookup.get_match(y-dy, from_x, to_x - 2 * radius, patch)
            if match is not None:
                return match

        return None


def line_scans(from_points: np.ndarray, from_image: np.ndarray, image_lookup, radius: int = 5, y_error_bound: int = 0):
    return [single_line_scan(x, y, from_image, image_lookup, radius, y_error_bound) for x, y in from_points]


def find_correspondence(img1, H1, img2, H2, grid_dist: int = 2, radius: int =4, y_error_bound: int = 0):
    """
    Find correspondences using epipolar geometry and SSD matching.
    I1, I2: Images as NumPy arrays (grayscale)
    F: Fundamental matrix (3x3)
    Returns: Dictionary mapping (x, y) in I1 to (x', y') in I2
    """

    # TODO: so much garbage... really need to clean this up


    # Generate the points in the first image, that we want to find the correspondence in the second.
    # We use points in a grid with distance 'jump', while ignoring the boundary of the image
    h1, w1 = img1.shape[:2]
    rows = np.arange(2, h1 // grid_dist - 2)[:, np.newaxis]
    cols = np.arange(2, w1 // grid_dist - 2)[np.newaxis, :]
    indices = np.swapaxes(np.dstack(np.meshgrid(cols, rows)), 0, 1).reshape(-1,2)

    # Prepare the corners of the second image inside which we look for the correspondent points
    h2, w2 = img2.shape[:2]
    # TODO: Should move to a smaller region to avoid the boundary of the image.
    #       Either do it here, of after applying the transformation
    corners2 = np.array([
        [0, 0,w2,w2, 0],
        [0,h2,h2, 0, 0],
        [1, 1, 1, 1, 1]
    ])

    img1, H1, img2, H2 = combined_coordinates(img1, H1, img2, H2)
    h1c, w1c = img1.shape[:2]

    wraped_indices = np.concatenate([indices * grid_dist, np.ones((indices.shape[0], 1))], axis=1) @ H1.T
    wraped_indices = (wraped_indices[:,:2] / wraped_indices[:, 2:3]).astype(int)
    in_boundary = ((wraped_indices[:,0] >= radius) & (wraped_indices[:,0] < w1c-radius) &
                   (wraped_indices[:,1] >= radius) & (wraped_indices[:,1] < h1c-radius))
    indices = indices[in_boundary]
    wraped_indices = wraped_indices[in_boundary]
    corners2 = H2 @ corners2
    corners2 = corners2[:2] / corners2[2:3]

    # At this point, both images have the same dimension, and they together with the points of interest and the corners
    # are in the same coordinate system.
    # More over the correspondence between the points should be on the same horizontal lines.

    image_lookup = ImageMatchLookup(img2, radius, corners2)

    correspondences = line_scans(wraped_indices, img1, image_lookup, radius, y_error_bound=y_error_bound)

    bad_positions = np.array([value is None for value in correspondences])
    bad_mask = np.zeros((h1 // grid_dist, w1 // grid_dist))
    bad_mask[indices[bad_positions,1], indices[bad_positions, 0]] = 1

    matches = np.zeros((h1 // grid_dist, w1 // grid_dist, 2))
    good_indices = indices[~bad_positions]

    matches[good_indices[:,1], good_indices[:,0]] = np.array([value for value in correspondences if value is not None])

    H2_inv = np.linalg.inv(H2)

    bad_points = indices[bad_positions]
    bad_x = bad_points[:,0]
    bad_y = bad_points[:,1]
    mask = np.zeros(shape=(h1 // grid_dist, w1 // grid_dist), dtype=int)
    mask[bad_y, bad_x] = 1
    mask[:2,:] = 1
    mask[-2:,:] = 1
    mask[:,2] = 1
    mask[:,-2:] = 1
    good_mask = 1-mask

    pre_counters = good_mask[2:,1:-1] + good_mask[:-2, 1:-1] + good_mask[1:-1,2:] + good_mask[1:-1,:-2]
    counters = mask.astype(int)
    counters[1:-1, 1:-1] *= pre_counters

    result = {tuple(pos): proj_prod(H2_inv, *value) for pos, value in zip(indices * grid_dist, correspondences) if value is not None}

    return result, mask[:, :, np.newaxis]

    # TODO: Future feature .....
    # index_to_counter = {tuple(idx): counter for idx, counter in zip(bad_points, counters[bad_y, bad_x])}
    # from vision.heap import MaxHeap
    # max_heap = MaxHeap(index_to_counter)
    #
    # def near_by_clique(neighbors, threshold = 15):
    #     distances = scipy.spatial.distance_matrix(neighbors, neighbors)
    #     if np.all(distances <= threshold):
    #         return neighbors
    #
    #     if len(neighbors) == 4:
    #         for subset in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
    #             if np.all(distances[np.ix_(subset, subset)] <= threshold):
    #                 return neighbors[subset, :]
    #
    #     return []
    #
    # while not max_heap.empty():
    #     (x,y), counter = max_heap.pop()
    #     if counter == 1:
    #         break
    #
    #     neighbors = []
    #     for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    #         match = result.get((grid_dist * (x + dx), grid_dist * (y + dy)), None)
    #         if match is not None:
    #             neighbors.append(match)
    #     print('----')
    #     print(f'{len(neighbors)=}')
    #     print(f'{counter=}')
    #     # assert counter == len(neighbors), f'problem with {x},{y}'
    #     neighbors = near_by_clique(np.array(neighbors), 15)
    #     if len(neighbors) == 0:
    #         continue
    #
    #     mean_x, _ = np.mean(neighbors, axis=0)
    #     mean_x = int(mean_x)
    #
    #     match = single_line_scan(*proj_prod(H1, x, y).astype(int),img1,image_lookup,radius,mean_x-15, mean_x+15)
    #     if match is None:
    #         continue
    #
    #     print('New point to add!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #     mask[y,x] = 0
    #     result[(grid_dist * x, grid_dist * y)] = proj_prod(H2_inv, *match)
    #     max_heap.try_update_value((x+1,y), 1)
    #     max_heap.try_update_value((x-1,y), 1)
    #     max_heap.try_update_value((x,y+1), 1)
    #     max_heap.try_update_value((x,y-1), 1)
    #
    # return result, mask[:,:,np.newaxis]

# </editor-fold>

def triangulate(p1, p2, calibration, rotation, position):
    """
    Compute 3D position from 2 projected positions
    """

    """
    
    Let: 
    - w be the actual position in 3D,
    - u, v the positions relative to the camera
    - u', v' their projections. Namely u=a_u*u', v=a_v*v' and u',v' has 1 in the z-coord.

    a_u C^{-1}u'    =    w    = P + a_v KC^{-1}v'
    
    We can compute 
    - dir1 = C^{-1}u'
    - dir2 = a_v KC^{-1}v'
    
    and then solve the linear system P = a_u*dir1 - a_v*dir2
    However, there is a lost of noise, so in general we don't expect the lines to actually intersect.
    Instead, we look for the place where they are the closest, namely we move from one to the other in a direction
    perpendicular to the directions of the lines 'perp'. Taking w to be the midpoint between the lines:
    
    a_u*dir1 + b*perp =  w  = P + a_v*dir2 - b*perp
    
    P = a_u*dir1- a_v*dir2 + 2b*perp 
    """
    # Convert to homogeneous coordinates
    dir1 = np.linalg.inv(calibration) @ np.array([p1[0], p1[1], 1])
    dir2 = rotation @ np.linalg.inv(calibration) @ np.array([p2[0], p2[1], 1])

    perp = np.cross(dir1, dir2)
    arr = np.array([dir1, -dir2, perp])
    (a_1, a_2, a_3) = position @ np.linalg.inv(arr)

    return a_1 * dir1 #+ (a_3/2) * perp

# <editor-fold desc=" ------------------------ Plots ------------------------">


EpipolarLine = NamedTuple('EpipolarLine', from_idx=int, to_idx=int, line=plt.Line2D, point=None | plt.Line2D)

class EpipolarPlotter:

    @staticmethod
    def compare_pictures(img1: np.ndarray, img2: np.ndarray, matrix: np.ndarray) -> 'EpipolarPlotter':
        plotter = EpipolarPlotter(img1, img2, matrix)
        plotter.update_with_mouse(
            plotter.add_epipolar_line(left_image=True, with_point=False, color='red'),
            mouse_click=False
        )
        plotter.update_with_mouse(
            plotter.add_epipolar_line(left_image=False, with_point=False, color='red'),
            mouse_click=False
        )
        return plotter

    def __init__(self, img1: np.ndarray, img2: np.ndarray, matrix: np.ndarray):
        self.matrices = [matrix, matrix.T]

        self.epipolar_lines = []
        self.move_plots = []
        self.click_plots = []

        self.fig, self.axes = plt.subplots(1, 2)
        self.fig.suptitle('Epipolar lines', fontsize=16)

        self.axes[0].imshow(img1)
        self.axes[1].imshow(img2)
        self.domains = [Rect.from_image(img1), Rect.from_image(img2)]


    def clear_all(self):
        for point, line in self.click_plots:
            point.set_data([],[])
            line.set_data([],[])
        for line in self.move_plots:
            line.set_data([],[])
        self.fig.canvas.draw()

    def add_epipolar_line(self, left_image: bool, with_point: bool, color: str):
        from_idx = 1 if left_image else 0
        to_idx = 1 - from_idx

        line = self.axes[to_idx].plot([], [], color=color, linestyle='-', lw=2)[0]
        point = None
        if with_point:
            point = self.axes[from_idx].plot([], [], color=color, marker='X', markersize=10)[0]

        epipolar_line = EpipolarLine(from_idx, to_idx, line, point)
        self.epipolar_lines.append(epipolar_line)
        return epipolar_line

    def update_with_mouse(self, epipolar_line: EpipolarLine, mouse_click: bool):
        """
        mouse_click = True for clicks, and False for mouse move
        """

        def mouse_update(event, epipolar_line: EpipolarLine):
            if event.inaxes == self.axes[epipolar_line.from_idx]:
                self.set_point(epipolar_line, event.xdata, event.ydata)

        mouse_event_type = 'button_press_event' if mouse_click else 'motion_notify_event'
        self.fig.canvas.mpl_connect(mouse_event_type, lambda event: mouse_update(event, epipolar_line))

    def set_point(self, epipolar_line, x, y):
        if epipolar_line.point is not None:
            epipolar_line.point.set_data([x], [y])

        a, b, c = np.array([x, y, 1]) @ self.matrices[epipolar_line.from_idx]
        p1, p2 = self.domains[epipolar_line.to_idx].line_endpoints(a, b, c)

        epipolar_line.line.set_data((p1[0], p2[0]), (p1[1], p2[1]))
        self.fig.canvas.draw()



def draw_epipolar_lines_on_axes(matrix: np.ndarray, fig, ax1, domain1: Rect, ax2, domain2:Rect):
    line1 = ax1.plot([], [], color='red', linestyle='-', lw=2)[0]
    line2 = ax2.plot([], [], color='red', linestyle='-', lw=2)[0]

    def on_mouse_move(event):
        if event.inaxes == ax1:
            x, y = event.xdata, event.ydata
            a, b, c = np.array([x,y,1]) @ matrix
            p1, p2 = domain2.line_endpoints(a, b, c)
            line2.set_data((p1[0], p2[0]), (p1[1], p2[1]))
            line1.set_data([],[])
            fig.canvas.draw()

        if event.inaxes == ax2:
            x, y = event.xdata, event.ydata
            a, b, c = np.array([x,y,1]) @ matrix.T
            p1, p2 = domain1.line_endpoints(a, b, c)
            line1.set_data((p1[0], p2[0]), (p1[1], p2[1]))
            line2.set_data([],[])
            fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)


def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray, matrix: np.ndarray):
    """
    Assumes that a pixel u from the first image corresponds to a pixel v from the second via the equation:
        (u1,u2,1) mat (v1, v2, 1)^T = 0

    Draw the two pictures side by side, and when pointing to a pixel on one of them draws the line on the second
    where the corresponding pixel can be.
    """

    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Epipolar lines', fontsize=16)

    axes[0].imshow(img1)
    axes[1].imshow(img2)

    draw_epipolar_lines_on_axes(matrix, fig,
                                axes[0], Rect.from_image(img1),
                                axes[1], Rect.from_image(img2))


    plt.show()


class LineScanVisual:

    def __init__(self):
        empty = np.zeros((10,10))
        self.fig = plt.figure(figsize=(15, 6))

        ax = plt.subplot2grid((3, 5), (0, 1))
        ax.set_title('Patch')
        self._patch_view   = ax.imshow(empty)

        ax = plt.subplot2grid((3, 5), (0, 2))
        ax.set_title('Diff')
        self._diff_view    = ax.imshow(empty)

        ax = plt.subplot2grid((3, 5), (0, 3))
        ax.set_title('To compare')
        self._compare_view = ax.imshow(empty)

        self._strip_ax = plt.subplot2grid((3, 5), (1, 0), colspan=5)
        self._strip_view = self._strip_ax.imshow(empty)

        self._graph = plt.subplot2grid((3, 5), (2, 0), colspan=5)
        self._fill_between = self._graph.fill_between([0,1],[1,1], [0,1], color='red', alpha=0.5)
        self._graph_view = self._graph.plot([1], [1])[0]
        self._min_view = self._graph.plot([], [], color='red', linestyle='-', lw=2)[0]
        self._min_height_view = self._graph.plot([], [], color='red', linestyle='-', lw=2)[0]

        self._graph.set_xlim([0, 2])

        strip = np.random.rand(11, 100)
        patch = strip[:, 50-5: 50+6]
        self.y = None
        self.set_data(radius = 5, patch=patch, strip=strip)

        self.fig.canvas.mpl_connect("motion_notify_event", lambda event: self.on_mouse_move(event))

    def set_compare(self, compare_img):
        self._compare_view.set_data(compare_img)

        diff_img = np.abs(np.abs(compare_img - self.patch))
        self._diff_view.set_data(diff_img)

        plt.draw()

    def on_mouse_move(self, event):
        if event.inaxes not in [self._graph, self._strip_ax]:  # Check if event is inside the desired axes
            return

        index = int(event.xdata) - self.radius
        if index < 0 or len(self.windows) <= index:
            return

        if self.y is not None:
            self.target_point.set_data([index+self.radius], [self.y])
            self.img_fig.canvas.draw_idle()

        self.set_compare(self.windows[index])

    def set_data(self, radius: int, patch: np.ndarray, strip:np.ndarray):
        assert patch.shape[:2] == (2*radius+1, 2*radius+1)
        assert strip.shape[0] == 2*radius+1
        self.radius = radius

        self.patch = patch.astype(np.int32)
        self._patch_view.set_data(self.patch)
        self._compare_view.set_data(self.patch)
        self._diff_view.set_data(self.patch)

        self._strip_view.set_data(strip)
        self._strip_view.set_extent([0, strip.shape[1], 0, strip.shape[0]])  # Force new width
        self._strip_ax.set_xlim(0, strip.shape[1])  # Adjust width dynamically
        # self._strip_ax.relim()
        # self._strip_ax.autoscale_view()
        self.windows = np.lib.stride_tricks.sliding_window_view(strip, (2 * radius + 1, 2 * radius + 1), axis=(0, 1))
        if len(strip.shape) == 3:
            self.windows = self.windows.transpose((0, 1, 3, 4, 2))
        self.windows = self.windows.squeeze(axis=0).astype(np.int32)

        self.norm_sqr = np.sqrt(np.sum((self.windows - self.patch) ** 2, axis=tuple(range(1, self.windows.ndim)))/(3*(2*radius+1)**2))
        x_range = np.arange(radius, radius+len(self.norm_sqr))
        self._graph_view.set_data(x_range, self.norm_sqr)
        # ax.fill_between(x, values, where=(values >= threshold), color='blue', alpha=0.5)
        self._graph.set_xlim([0, len(self.norm_sqr) + 2*radius])
        self._graph.set_ylim([0, 256])

        x = np.argmin(self.norm_sqr) + radius
        self.set_compare(self.windows[x-radius])
        if self.y is not None:
            self.min_point.set_data([x], [self.y])
            self.img_fig.canvas.draw_idle()
        y = np.min(self.norm_sqr)

        self._fill_between.remove()
        self._fill_between = self._graph.fill_between(x_range, self.norm_sqr, where=(self.norm_sqr < y+MIN_VALUE_GAP), color='red', alpha=0.5)

        self._min_view.set_data((x,x), (0,256))
        self._min_height_view.set_data((0, len(self.norm_sqr) + 2*radius), (y,y))

        self.fig.canvas.draw_idle()

    def from_images(self, img1: np.ndarray, img2: np.ndarray, radius: int = 5, mask1: Optional[np.ndarray] = None):

        self.img_fig, axes = plt.subplots(1, 2)
        if mask1 is None:
            axes[0].imshow(img1)
        else:
            axes[0].imshow((img1*mask1).astype(int))
        axes[1].imshow(img2)


        click_line = axes[1].plot([], [], color='red', linestyle='-', lw=2)[0]
        mouse_line = axes[1].plot([], [], color='green', linestyle='-', lw=2)[0]
        click_point = axes[0].plot([], [], color='red', marker='X', markersize=10)[0]
        self.target_point = axes[1].plot([], [], color='blue', marker='X', markersize=10)[0]
        self.min_point = axes[1].plot([], [], color='red', marker='X', markersize=10)[0]

        # draw_epipolar_lines(np.array([[0,0,0],[0,0,1],[0,-1,0]]), fig,
        #                     axes[0], Rect.from_image(img1),
        #                     axes[1], Rect.from_image(img2))

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        def on_press(event):
            if event.inaxes != axes[0]:
                return

            x, y = int(event.xdata), int(event.ydata)
            if x<radius or w1-radius<=x or y<radius or h1-radius<=y or h2-radius<=y:
                pass

            self.y = y

            click_line.set_data((0, w2), (y, y))
            click_point.set_data([x], [y])

            patch = img1[y-radius: y+radius+1, x-radius:x+radius+1]
            strip = img2[y-radius: y+radius+1, :]
            self.set_data(radius, patch, strip)

        self.img_fig.canvas.mpl_connect('button_press_event', on_press)

        def on_mouse_move(event):
            if event.inaxes == axes[0]:
                x, y = event.xdata, event.ydata
                mouse_line.set_data((0,w2),(y,y))
            else:
                mouse_line.set_data([],[])
            self.img_fig.canvas.draw()

        self.img_fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)



def show_bad_points(image: np.ndarray, jump:int, bad_point_mask:np.ndarray):
    h,w = image.shape[:2]
    jumped_img = image[0: h: jump,0: w:jump,:]
    fig = plt.figure()
    plt.imshow(jumped_img * bad_point_mask)
    fig.suptitle('Bad points', fontsize=16)
    plt.show()
# </editor-fold>


def find_relative_position(fundamental_mat: np.ndarray, camera_mat: np.ndarray, matchings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the relative position (3,) and rotation (3,3) for the given parameters
    """
    essential = camera_mat.T @ fundamental_mat @ camera_mat
    (skew, _), pos_K, essential = skew_ortho_decomposition(essential)
    translation = np.array([-skew[1, 2], skew[0, 2], -skew[0, 1]])
    pos_P = (translation, -translation)

    best_counter = -1
    best_param = None
    best_index = 0

    index = 1

    for P in pos_P:
        for K in pos_K:
            counter = 0

            for (x, y), (x2, y2) in matchings:
                pos = triangulate((x, y), (x2, y2), camera_mat, K, P)
                pos1 = pos
                pos2 = K.T @ (pos - P)
                if pos2[2] > 0 and pos1[2]>0:
                    counter += 1
            print(f'{index}. Counted {counter}/{len(matchings)} points in front of both cameras')
            if best_counter < counter:
                best_index = index
                best_counter = counter
                best_param = (P, K)
            index += 1

    print(f'Chose {best_index}')
    return best_param


def combined_coordinates(img1: np.ndarray, transform1: np.ndarray, img2: np.ndarray, transform2: np.ndarray):
    """
    Update both images and transforms to be in the same coordinate system, and in the positive quadrant
    """

    domain1 = Rect.from_image(img1).apply_proj(transform1)
    transform1 = translation_mat(-domain1.x, 0) @ transform1
    # domain1 = domain1.translate(-domain1.x, 0)

    domain2 = Rect.from_image(img2).apply_proj(transform2)
    transform2 = translation_mat(-domain2.x, 0) @ transform2
    # domain2 = domain2.translate(-domain2.x, 0)

    img1, domain1 = apply_proj(transform1, img1)
    img2, domain2 = apply_proj(transform2, img2)

    domain = domain1.union(domain2)
    img1 = change_domain(img1, domain1, domain)
    img2 = change_domain(img2, domain2, domain)

    translation = translation_mat(-domain.x, -domain.y)
    transform1 = translation @ transform1
    transform2 = translation @ transform2

    # domain = domain.translate(-domain.x, -domain.y)

    return img1, transform1, img2, transform2


def compute_point_cloud(image, correspondences, camera_mat, rel_rotation, rel_position):

    pixel_position = []
    points = []
    colors = []
    for pos1, pos2 in correspondences.items():
        pixel_position.append(pos1)
        pos_3d = triangulate(pos1, pos2, camera_mat, rel_rotation, rel_position)
        points.append(pos_3d)
        x, y = pos1
        colors.append(np.mean(image[y-1:y+2, x-1: x+2, :],axis=(0,1)))
    return np.array(pixel_position), np.array(points), np.array(colors)


def compute_depth_map2(h, w, jump, correspondences, camera_mat, rel_rotation, rel_position):
    depth_map = np.zeros(shape=(h // jump, w // jump))
    min_value = float('inf')
    max_value = float('-inf')
    for pos1, (wp1, wp2, pos2) in correspondences.items():
        pos_3d = triangulate(pos1, pos2, camera_mat, rel_rotation, rel_position)
        z = pos_3d[2]
        if min_value > z:
            min_value = z
        if max_value < z:
            max_value = z
        depth_map[pos1[1] // jump, pos1[0] // jump] = pos_3d[2]
    return depth_map


def good_points_mask(points: np.ndarray, from_per: float, to_per: float):
    """
    points: n x 3 array
    1. Removes points with nonpositive z-value (behind camera)
    2. Keep only points in a given percentile (from the positive z points)

    Return a bool mask for the good points remaining
    """
    positive_points_mask = points[:, 2] > 0
    print(f'There are {sum(positive_points_mask)}/{len(points)} points in front of the camera')

    positive_points = points[positive_points_mask]

    lower_bound = np.percentile(positive_points, 1, axis=0)
    upper_bound = np.percentile(positive_points, 99, axis=0)

    percentile_mask = \
        ((points[:, 0] >= lower_bound[0]) & (points[:, 0] <= upper_bound[0]) &
         (points[:, 1] >= lower_bound[1]) & (points[:, 1] <= upper_bound[1]) &
         (points[:, 2] >= lower_bound[2]) & (points[:, 2] <= upper_bound[2]))

    return positive_points_mask & percentile_mask



def generate_point_cloud(*images: np.ndarray, camera_mat:np.ndarray, jump: int = 2, stretch_x: float = 1, shear_value: float = 0) -> np.ndarray:
    """
    Stitch the images together. Assumes that all the images intersect in a meaningful way the
    first image (lots of interesting keypoints in the intersection).
    """

    # Create matching
    image_system = ImageSystem(list(images))
    matchings = image_system.matching(from_index=0, to_index=1)

    fundamental = least_median_fundamental(matchings)

    # EpipolarPlotter.compare_pictures(images[0], images[1], fundamental)
    # plt.show()
    # exit(0)
    H0, H1 = y_align_images(Rect.from_image(images[0]), Rect.from_image(images[1]), fundamental)

    x_distort = np.array([
        [stretch_x,shear_value,0],
        [0,1,0],
        [0,0,1]])
    H0 = x_distort @ H0
    img0, H0, img1, H1 = combined_coordinates(images[0], H0, images[1], H1)

    # EpipolarPlotter.compare_pictures(img0, img1, np.array([[0,0,0],[0,0,1],[0,-1,0]]))
    # plt.show()
    # exit(0)

    correspondences, bad_points_mask = find_correspondence(images[0], H0, images[1], H1, grid_dist = jump, radius = 4)

    # draw_corrsepondence(images[0], images[1], correspondences, jump)
    # exit(0)

    # resized_bad_points = cv2.resize(bad_points_mask.astype(float), dsize=(0,0), fx=jump, fy=jump)
    # warped_bad_points, bp_domain = apply_proj(H0, resized_bad_points)
    # warped_bad_points = change_domain(warped_bad_points, bp_domain, Rect.from_image(img0))
    # line_scan_visual = LineScanVisual()
    # line_scan_visual.from_images(img1=img0, img2=img1, radius = 4, mask1=warped_bad_points[:,:,np.newaxis])
    # plt.show()
    # exit(0)

    rel_position, rel_rotation = find_relative_position(fundamental, camera_mat, matchings)
    rel_position /= np.linalg.norm(rel_position)
    print(f'{rel_position=}')

    pixel_position, points, colors = compute_point_cloud(images[0], correspondences, camera_mat, rel_rotation, rel_position)

    mask1 = good_points_mask(points, 1, 99)
    pixel_position = pixel_position[mask1]
    points = points[mask1]
    colors = colors[mask1]

    print(f'There are {len(points)} points remained')

    return pixel_position, points, colors




