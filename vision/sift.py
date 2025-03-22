# https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5
# https://www.ipol.im/pub/art/2014/82/article_lr.pdf
# https://www.youtube.com/watch?v=ram-jbLJjFg&t=286s&ab_channel=FirstPrinciplesofComputerVision


import copy
import cv2
from dataclasses import dataclass, field
from functools import cache
import math
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter, minimum_filter
from time import time
from typing import List, Tuple, NamedTuple, Optional, ClassVar


@dataclass
class KeyPoint:
    _obj_created: ClassVar[int] = 0

    position: np.ndarray = field(default_factory=lambda:np.zeros(shape=(2,))) # row, col in image
    orientation: float = 0            # in [0,1], as part of the full circle.
    octave: int = 0
    scale: int = 0
    sigma: float = 0
    value: float = 0
    delta: float = 0
    hist: np.ndarray = field(default_factory=lambda:np.zeros(shape=(1,)))
    integer: np.ndarray = field(default_factory=lambda:np.zeros(shape=(1,)))
    mod: np.ndarray = field(default_factory=lambda:np.zeros(shape=(1,)))
    hist: np.ndarray = field(default_factory=lambda:np.zeros(shape=(1,)))
    id: int = 0

    def __post_init__(self):
        self.update_id()

    def update_id(self):
        self.id = KeyPoint._obj_created
        KeyPoint._obj_created += 1

    def copy(self) -> 'KeyPoint':
        copy_version = copy.copy(self)
        copy_version.update_id()
        return copy_version

_idx_positions = np.dstack([np.indices((500, 500))[0], np.indices((500, 500))[1]])

# <editor-fold desc=" ------------------------ Gaussians and resolutions ------------------------">

@cache
def gaussian_blur1d(sigma: float) -> np.ndarray: # shape (ceil(2*sigma),)
    radius = int(np.ceil(3.24 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()

def half_resolution(image:np.ndarray) -> np.ndarray:
    h,w = image.shape
    if h%2 == 1:
        image = image[:-1,:]
    if w%2 == 1:
        image = image[:,:-1]
    return image[::2,::2]

def gaussian_blur_image(image:np.ndarray, num_scales: int, num_octaves: int) -> List[np.ndarray]:
    """
    returns a list of octaves. Each octave an array of shape :   (num_scales + 3) x h x w
    """
    def generate_octave(first_image:np.ndarray):
        # We try to keep the image at octave o and blur scale s with the
        # Gaussian Blur with standard variation:
        #           2*sigma_min * 2^(o-2) * 2^(s/num_scales)
        # Where sigma_min is the standard variation of the original image (which we think of as 0.8)
        # The 2^(o-2) will be produced (apparently?) by the resolution doubling \ halfing, so we only need to take
        # care of the rest.
        # We use the fact that repeated Gaussians has the property:
        #           G_{sigma_1^2} * G_{sigma_2^2} = G_{sigma_1^2+sigma_2^2}.

        four_power = 1
        ratio = 4 ** (1/num_scales)
        coef = sigma_min/delta_min
        sigmas = []
        for scale in range(num_scales + 2):
            sigmas.append(coef * np.sqrt((ratio-1)*four_power))
            four_power *= ratio

        blurred_images = [first_image]
        for sig in sigmas:
            blurred_images.append(blur_by(blurred_images[-1], sig))

        return blurred_images

    # Generate first octave
    octaves = [generate_octave(image)]

    # Generate rest
    for octave in range(num_octaves-1):
        image = half_resolution(octaves[-1][num_scales])
        octaves.append(generate_octave(image))

    return octaves

def blur_by(image:np.ndarray, sigma: float):
    """
    2D Gaussian Blur
    """
    # blur_kernel = gaussian_blur1d(sigma)
    # rows_blur = convolve1d(input=image,     weights=blur_kernel, axis=1, mode='mirror')
    # full_blur = convolve1d(input=rows_blur, weights=blur_kernel, axis=0, mode='mirror')
    # return full_blur
    return cv2.GaussianBlur(image, (0,0), sigmaX=sigma, sigmaY=sigma)

# </editor-fold>

# <editor-fold desc=" ------------------------ Generate Image Data ------------------------">

def initial_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_change = (1/delta_min) * np.sqrt(sigma_min**2 - sigma_in**2)
    blur_kernel = gaussian_blur1d(sigma_change)

    rows_blur = convolve1d(input=image,     weights=blur_kernel, axis=1, mode='mirror')
    image     = convolve1d(input=rows_blur, weights=blur_kernel, axis=0, mode='mirror')
    return image

def image_diff(images: np.ndarray) -> np.ndarray:
    return images[1:]-images[:-1]

# </editor-fold>

class ImageData:

    def __init__(self, image: np.ndarray, num_scales: int, num_octaves: int):
        self.image = image
        self.base_image = initial_image(image)
        self.gaussian_images = gaussian_blur_image(self.base_image, num_scales, num_octaves)
        der_conv = np.array([1, 0, -1])
        self.gradients = [[np.array([convolve1d(arr, der_conv, axis=0, mode='nearest'),      # dy
                                     convolve1d(arr, der_conv, axis=1, mode='nearest')])     # dx
                            for arr in octave]
                            for octave in self.gaussian_images]
        self.dog_images = [image_diff(np.array(elem)) for elem in self.gaussian_images]

    def dim_at(self,octave_idx: int, scale_idx: int):
        return np.array(self.gradients[octave_idx-1][scale_idx].shape[1:])


Position = NamedTuple('Position',
                      integer=Tuple[int,int,int],   # scale, row, col
                      mod=Tuple[float,float,float],
                      value=float,
                      )

# <editor-fold desc=" ------------------------ Keypoints positions ------------------------">

def box_extrema_points(arr: np.ndarray, threshold: float = -1, ignore_border = (1,1,1)):
    """
    In the 3D array 'arr' look for local minima and maxima in a 3x3x3 windows (no points on the boundary).
    Only keep those that are above the threshold in absolute value.
    """
    local_max = (arr == maximum_filter(arr, size=(3, 3, 3)))
    local_min = (arr == minimum_filter(arr, size=(3, 3, 3)))
    extrema_points = local_min | local_max
    if threshold > 0:
        extrema_points &= (np.abs(arr) >= threshold)

    # Remove points on the border
    border = np.ones(arr.shape, dtype=bool)
    if ignore_border[0] > 0:
        border[:ignore_border[0] , :, :] = False
        border[-ignore_border[0]:, :, :] = False
    if ignore_border[1] > 0:
        border[:, :ignore_border[1] , :] = False
        border[:, -ignore_border[1]:, :] = False
    if ignore_border[2] > 0:
        border[:, :, :ignore_border[2] ] = False
        border[:, :, -ignore_border[2]:] = False
    extrema_points &= border

    return extrema_points

def derivatives3(cube:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the hessian and gradient of the center of the pixel cube
    """
    grad = np.array([
        cube[2, 1, 1] - cube[0, 1, 1],
        cube[1, 2, 1] - cube[1, 0, 1],
        cube[1, 1, 2] - cube[1, 1, 0],
    ]) / 2

    h11 = cube[2, 1, 1] - 2 * cube[1, 1, 1] + cube[0, 1, 1]
    h22 = cube[1, 2, 1] - 2 * cube[1, 1, 1] + cube[1, 0, 1]
    h33 = cube[1, 1, 2] - 2 * cube[1, 1, 1] + cube[1, 1, 0]

    h12 = (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1]) / 4
    h13 = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0]) / 4
    h23 = (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0]) / 4

    hessian = np.array([
        [h11, h12, h13],
        [h12, h22, h23],
        [h13, h23, h33],
    ])
    return grad, hessian

def keypoint_interpolation(arr:np.ndarray, position: np.ndarray, ignore_border=(1,5,5)) -> Optional[Position]:
    for _ in range(5):
        if any(index < border or size-border <= index for index, size, border in zip(position, arr.shape, ignore_border)):
            return None

        i,j,k = position
        pixel_cube = arr[i-1:i+2,j-1:j+2,k-1:k+2].astype('float32') / 255
        # Recall that if
        #               f(x_0+h) ~ f(x_0) + f'(x_0)h+(f''(x_0)/2)*h^2
        # Then we expect the minimum (as function of h) to be at
        #               h = - f'(x_0)/f''(x_0)
        grad, hessian = derivatives3(pixel_cube)
        alpha = -np.linalg.inv(hessian) @ grad
        # alpha = -np.linalg.lstsq(hessian, grad, rcond=None)[0]

        if np.max(np.abs(alpha))<0.50:
            # Value at new position, using the approximation from above
            return Position(
                integer=tuple(int(e) for e in position), mod=tuple(float(alph) for alph in alpha),
                value=pixel_cube[1,1,1] + 0.5 * alpha @ grad)
        position += np.round(alpha).astype(dtype=int)
    return None

def edge_ratio(arr: np.ndarray, position: Tuple[int, int, int]) -> float:
    i, j, k = position
    if (1, 169, 268) == (i,j,k):
        print('stop')
    neigh = arr[i,j-1:j+2,k-1:k+2].astype('float32') / 255

    h11 = neigh[2, 1] - 2*neigh[1, 1] + neigh[0, 1]
    h22 = neigh[1, 2] - 2*neigh[1, 1] + neigh[1, 0]
    h12 = (neigh[2, 2] - neigh[2, 0] - neigh[0, 2] + neigh[0, 0])/4
    trace = h11 + h22
    det = h11 * h22 - h12 * h12

    return np.abs(trace*trace / det)

def validate_keypoint_position(
        arr: np.array, position: np.ndarray, abs_threshold:float, edge_threshold:float ):
    """
    position is an array of size 3 of integers
    """
    interpolated_position = keypoint_interpolation(arr, position)
    if (interpolated_position is not None and
            abs(interpolated_position.value) >= abs_threshold and
            edge_ratio(arr=arr, position=interpolated_position.integer) <= edge_threshold):
        return interpolated_position
    return None

# </editor-fold>


def convert_to_keypoint(position: Position, delta:float, num_scales: int, octave:int) -> KeyPoint:
    global delta_min, sigma_min
    pos = [int_part + mod_part for int_part, mod_part in zip(position.integer, position.mod)]
    sigma = (delta / delta_min) * sigma_min * 2 ** (pos[0] / num_scales)
    return KeyPoint(
        position = np.array([pos[2]*delta, pos[1]*delta]),  # col,row in original image res
        sigma = 2 * sigma,                                  # 2 * sigma     TODO: remove the 2?
        value = abs(position.value),                        # approximate value
        scale = position.integer[0],                        # scale index in octave
        delta = delta,                                      # distance between pixels in image
        integer = np.array(position.integer[1:]),           # int position in image
        mod = np.array(position.mod[1:]),                   # mod position in image
        octave = octave,                                    # in 1,2,...,num_octaves
    )

# <editor-fold desc=" ------------------------ Orientations ------------------------">

def compute_orientation(
        center: np.ndarray, relative_unit: float, cur_gradients:np.ndarray,
        num_bins = 36, peak_ratio = 0.8) -> List[KeyPoint]:

    radius = int(np.round(4.5 * relative_unit))    # TODO: Can add here a parameter to control the radius

    img_dim = np.array(cur_gradients.shape[1:])
    # Remove the boundaries, since the gradients are not well-defined there.
    y_low, x_low = np.maximum(1, center-radius)
    y_high, x_high = np.minimum(img_dim-2, center+radius)

    dy, dx = cur_gradients[:, y_low:y_high+1, x_low:x_high+1]

    grad_orientation = np.arctan2(dy, dx)/(2*np.pi)
    bin_positions = np.round(-num_bins * grad_orientation).astype(int) # TODO: There is a minus here?
    bin_positions = bin_positions % num_bins

    # indices relative to the center
    cur_positions = _idx_positions[0: y_high-y_low+1, 0: x_high-x_low+1, :] + np.array([y_low, x_low]) - center
    weight_factor = -1 / (4.5 * relative_unit ** 2)
    weight = np.exp(weight_factor * np.sum(cur_positions**2, axis=2)) * np.sqrt(dx**2 + dy**2)

    histogram = np.bincount(bin_positions.reshape(-1), weights=weight.reshape(-1), minlength=num_bins)
    # A bit of smoothening
    histogram = convolve1d(histogram, weights=np.array([1,4,6,4,1])/16, mode='wrap')

    orientation_max = np.max(histogram)
    local_max_peaks = np.where(
        (histogram > np.roll(histogram, 1)) &  # local maxima
        (histogram > np.roll(histogram, -1)) &
        (histogram > orientation_max * peak_ratio)  # and big enough
    )[0]

    orientations = []
    for peak_index in local_max_peaks:
        left_value  = histogram[(peak_index - 1) % num_bins]
        peak_value  = histogram[ peak_index ]
        right_value = histogram[(peak_index + 1) % num_bins]
        interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value))
        interpolated_peak_index %= num_bins
        orientations.append(1-interpolated_peak_index / num_bins)

    return orientations

s2 = math.sqrt(2)

def bin_sum_interpolation(
        bin_positions: np.ndarray, bin_values: np.ndarray,
        bin_size1: int, bin_size2: int, bin_size3: int):
    """
    Collect the bin values into the bin positions in a bin_size1 x bin_size2 x bin_size3 bins.
    The bin_positions n x 3 don't need to be integers, and we use trilinear interpolation
    to add their corresponding values to the nearest bins.
    The first two bin dimensions are standard, while the third is cyclic.
    """
    bin_pos_floor = np.floor(bin_positions).astype(int)
    bin_pos_frac = bin_positions - bin_pos_floor
    bin_pos_floor[:, 2] %= bin_size3  # The third position is cyclic

    # Trilinear interpolation:
    # 1D distance from the two nearest integers
    bin_coefs = np.stack([1 - bin_pos_frac, bin_pos_frac], axis=-1)
    bin_coefs0 = bin_coefs[:, 0, :]
    bin_coefs1 = bin_coefs[:, 1, :]
    bin_coefs2 = bin_coefs[:, 2, :]
    # 3D distance: product of distances in each axis
    dist_prod = bin_coefs0[:, :, None, None] * bin_coefs1[:, None, :, None] * bin_coefs2[:, None, None, :]

    final_values = np.array(bin_values)[:, None, None, None] * dist_prod

    # First two dimensions are increased by 2 to account for border effects
    # Third dimension is cyclic
    histogram_tensor = np.zeros((bin_size1 + 2, bin_size2 + 2, bin_size3))
    for (pos0, pos1, pos2), final_value in zip(bin_pos_floor, final_values):
        if pos2 < bin_size3 - 1:
            histogram_tensor[pos0 + 1:pos0 + 3, pos1 + 1:pos1 + 3, pos2: pos2 + 2] += final_value
        else:
            histogram_tensor[pos0 + 1:pos0 + 3, pos1 + 1:pos1 + 3, [bin_size3 - 1, 0]] += final_value

    return histogram_tensor[1:-1, 1:-1, :]  # Remove histogram borders

def create_descriptors(
        keypoints:List[KeyPoint], image_data, n_hist: int = 4, num_angle_bins = 8,
        descriptor_max_value=0.2, float_tolerance = 1e-7):
    """
    Generate descriptors for each keypoint
    """
    descriptors = []
    for keypoint in keypoints:
        square_width = 1.5 * keypoint.sigma/keypoint.delta

        radius = int(round(square_width * (n_hist+1) * s2 * 0.5))
        img_dim = image_data.dim_at(keypoint.octave, keypoint.scale)

        # Compute the data in the window around the keypoint.
        # Remove the boundary of the image, since the gradients are not well-defined there.
        y_low, x_low = np.maximum(1, keypoint.integer - radius)
        y_high, x_high = np.minimum(img_dim-2, keypoint.integer + radius)

        # indices relative to the center
        cur_positions = _idx_positions[0: y_high - y_low + 1, 0: x_high - x_low + 1, :] + np.array([y_low, x_low]) - keypoint.integer

        # bin positions in space
        cos_angle = np.cos(-keypoint.orientation * 2 * np.pi)
        sin_angle = np.sin(-keypoint.orientation * 2 * np.pi)
        rot_positions = np.tensordot(cur_positions, np.array([[cos_angle,-sin_angle],
                                                              [sin_angle, cos_angle]]), axes=1)
        rot_positions /= square_width
        bin_position = rot_positions + 0.5 * n_hist - 0.5

        # bin position for angle, and magnitude
        cur_gradients = image_data.gradients[keypoint.octave-1][keypoint.scale]
        dy, dx = cur_gradients[:, y_low: y_high + 1, x_low: x_high + 1]

        gradient_orientation = (np.arctan2(dy, dx)/(2*np.pi)) % 1
        angle_bin = (keypoint.orientation - gradient_orientation) * num_angle_bins  # TODO: do I need another mod num_bins?

        #                       total length of the square
        weight_multiplier = -2 / ((n_hist * square_width) ** 2)
        weight = np.exp(weight_multiplier * np.sum(cur_positions**2, axis=2))
        weight *= np.sqrt(dx**2 + dy**2)

        bin_positions = []
        bin_values = []

        for row in range(y_high - y_low + 1):
            for col in range(x_high - x_low + 1):
                pos = bin_position[row, col]
                if not (-1 < pos[0] < n_hist and -1 < pos[1] < n_hist):
                    continue
                bin_positions.append((pos[0], pos[1], angle_bin[row, col]))
                bin_values.append(weight[row, col])

        histogram_tensor = bin_sum_interpolation(
            np.array(bin_positions), np.array(bin_values),
            n_hist, n_hist, num_angle_bins)

        descriptor_vector = histogram_tensor.flatten()  # Remove histogram borders

        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)

        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
        keypoint.hist = descriptor_vector

    return np.array(descriptors, dtype='float32')

# </editor-fold>


# image_name = 'stitch2_10'
# img_path = f'images/{image_name}.jpg'
# image = mpimg.imread(img_path)
# image = np.mean(image, axis=2)

sigma_min  = 0.8
delta_min  = 0.5
sigma_in   = 0.5
num_scales = 3

def process_image(image: np.ndarray):
    start_time = time()
    contrast_threshold = 0.04
    threshold = np.floor(0.5 * contrast_threshold / num_scales * 255)
    image_border_width = 5
    eigenvalue_ratio = 10
    edge_threshold = ((eigenvalue_ratio + 1) ** 2)/eigenvalue_ratio

    base_image = initial_image(image)
    num_octaves = int(round(math.log2(min(base_image.shape)) - 1))
    image_data = ImageData(image, num_scales=num_scales, num_octaves=num_octaves)

    cur_delta = delta_min
    final_keypoints = []
    for oct_idx, oct_images in enumerate(image_data.dog_images):
        # Find local maximum
        extrema_points = box_extrema_points(oct_images, threshold, ignore_border=(1, image_border_width, image_border_width))

        for position in np.argwhere(extrema_points):
            # interpolated quadratically, and make sure that it is "valid" (not too edgy...)
            interpolated_position = validate_keypoint_position(
                oct_images, position, contrast_threshold / num_scales, edge_threshold)
            if interpolated_position is None:
                continue
            keypoint = convert_to_keypoint(
                interpolated_position, delta=cur_delta, num_scales=num_scales, octave=oct_idx + 1)

            # The bigger the sigma (zoom) is, the larger the units become,
            # and the bigger the delta (1/resolution) is, we need to look at fewer pixels
            relative_unit = keypoint.sigma * delta_min / keypoint.delta
            center = keypoint.integer

            oritentations = compute_orientation(
                center, relative_unit, image_data.gradients[keypoint.octave - 1][keypoint.scale])
            for orientation in oritentations:
                keypoint = keypoint.copy()
                keypoint.orientation = orientation
                final_keypoints.append(keypoint)

        cur_delta *= 2

    my_decriptors = create_descriptors(final_keypoints, image_data)
    print(f'Inside took {time() - start_time} seconds')

    return final_keypoints, my_decriptors, image_data




