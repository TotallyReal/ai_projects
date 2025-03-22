import cv2
import numpy as np
from typing import List, Tuple, Optional
import numbers

# ============================================================================= #
#
# See the panorama.ipynb file to see how to use this file
#
# ============================================================================= #


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
    # Since |WA|=tr(A^T W^T*W A), then using the spectral decomposition of the positive semi-definite matrix W^TW,
    # we conclude that A is the eigenvector of W^T*W corresponding to the smallest eigenvalue.
    eigenvalues, eigenvectors = np.linalg.eigh(arr.T @ arr)
    return eigenvectors[:, 0].reshape(3, 3)

def point_matching(
        bf_matcher: cv2.DescriptorMatcher,
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


Point = Tuple[float, float]    # x, y

class Rect:

    @staticmethod
    def from_image(image: np.ndarray):
        h, w = image.shape[:2]
        return Rect(0, 0, w, h)

    def __init__(self, x:int, y:int, width:int, height:int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return f'[({self.x},{self.y}), +({self.width},{self.height})]'

    def __repr__(self):
        return self.__str__()

    def translate(self, dx, dy) -> 'Rect':
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def in_local(self, points: List[Tuple[int, int]] | Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        transform the points to the coordinates inside the rectangle (so self.x, self.y becomes 0,0)
        """
        if len(points) == 2 and all(isinstance(elem, numbers.Number) for elem in points):
            x, y = points
            return x-self.x, y-self.y
        return [(x-self.x, y-self.y) for x, y in points]

    def from_local(self, points: List[Tuple[int, int]] | Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        transform the points from the coordinates inside the rectangle (so 0,0 becomes self.x, self.y)
        """
        if len(points) == 2 and all(isinstance(elem, numbers.Number) for elem in points):
            x, y = points
            return x+self.x, y+self.y
        return [(x+self.x, y+self.y) for x, y in points]

    def relative_to(self, domain: 'Rect') -> 'Rect':
        return self.translate(-domain.x, -domain.y)

    def intersect(self, other) -> 'Rect':
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2, y2 = min(self.x + self.width, other.x + other.width), min(self.y + self.height, other.y + other.height)
        if x1 < x2 and y1 < y2:
            return Rect(x1, y1, x2 - x1, y2 - y1)
        return None  # No intersection

    def union(self, other) -> 'Rect':
        # Smallest rectangle containing both rectangles
        x1, y1 = min(self.x, other.x), min(self.y, other.y)
        x2, y2 = max(self.x + self.width, other.x + other.width), max(self.y + self.height, other.y + other.height)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def as_slices(self, x_first:bool=False) -> Tuple[slice, slice]:
        x_slice = slice(self.x, self.x+self.width)
        y_slice = slice(self.y, self.y+self.height)
        if x_first:
            return x_slice, y_slice
        return y_slice, x_slice

    def line_endpoints(self, a: float, b: float, c: float) -> Tuple[Point, Point]:
        """
        returns two endpoints of a segment in the line ax+by+c=0 which contains its intersection with this rectangle
        the rectangle.
        """
        if b == 0: # vertical line
            x = -c/a
            return (x, self.y), (x, self.y + self.height)
        else:
            # y = -(ax+c)/b
            return (self.x, -(a*self.x+c)/b), (self.x+self.width, -(a*(self.x+self.width)+c)/b)

    def corners(self, projective: bool = False) -> np.ndarray:
        """
        returns the corners of this rect in an anticlock wise order.
        The corners are a 2 x 4 array. If projective is true, then add a line of 1's.
        """
        x1, x2 = self.x, self.x + self.width
        y1, y2 = self.y, self.y + self.height
        if projective:
            return np.array([
                [x1, x2, x2, x1],
                [y1, y1, y2, y2],
                [ 1,  1,  1,  1]
            ])
        else:
            return np.array([
                [x1, x2, x2, x1],
                [y1, y1, y2, y2]
            ])

    def apply_proj(self, proj: np.ndarray) -> 'Rect':
        """
        returns the smallest Rect which contains the image of this rect under the projective map
        """
        corners = self.corners(projective=True)
        # x1, x2 = self.x, self.x + self.width
        # y1, y2 = self.y, self.y + self.height
        # corners = np.array([
        #     [x1, x1, x2, x2],
        #     [y1, y2, y1, y2],
        #     [1, 1, 1, 1]
        # ])
        corners = proj @ corners
        corners = corners[:2] / corners[2:3]
        corners = np.round(corners).astype(int)

        min_x, min_y = np.min(corners, axis=1)
        max_x, max_y = np.max(corners, axis=1)
        return Rect(min_x, min_y, max_x - min_x, max_y - min_y)



def apply_proj(proj: np.ndarray, image: np.ndarray, domain: Optional[Rect] = None) -> Tuple[np.ndarray, Rect]:
    """
    Considering 'image' as a function f on R^2 (with values in R for gray and R^3 for RGB), We would like
    to return the function :
         (proj*f) (u,v) := f( proj^-1 (u,v) )

    There are two problems:
        1. f is defined on integers, while proj^-1 (u,v) are not necessarily integers. For that we use the
           cv2.warpPerspective interpolation.
        2. Both f and the returned function are represented using arrays, so they are defined on a rectangle
           from (0,0) to some positive (h,w). We only compute (proj*f) in the given domain_x=[x0,x1], domain_y=[y0,y1]
           so that:
               result[i,j] = (proj*f) (x0+i, y0+j) = f( floor(proj^-1 (x0+i, y0+j)) )
    """
    if domain is None:
        domain = Rect.from_image(image)

    new_domain = domain.apply_proj(proj)

    translation_to = np.array([
        [1,0,-new_domain.x],
        [0,1,-new_domain.y],
        [0,0,1]
    ])
    translation_from = np.array([
        [1,0,domain.x],
        [0,1,domain.y],
        [0,0,1]
    ])


    return cv2.warpPerspective(
        image, translation_to @ proj @ translation_from, (new_domain.width, new_domain.height)), new_domain



def change_domain(image: np.ndarray, image_domain: Rect, to_domain: Rect):
    domain = image_domain.intersect(to_domain)

    shape = (to_domain.height, to_domain.width)
    if len(image.shape) == 3:   # For RGB
        shape += (3,)
    result = np.zeros(shape=shape, dtype=image.dtype)

    if domain is not None:
        result[domain.relative_to(to_domain).as_slices()] = image[domain.relative_to(image_domain).as_slices()]

    return result


class ImagePart:

    def __init__(self, image: np.ndarray, transform: np.ndarray):
        self.image = image
        self.transform = transform
        self.transformed_image, self.domain = apply_proj(self.transform, self.image)

        # create weights
        self.h, self.w = image.shape[:2]

        rows = np.arange(0,1,1/self.h)
        cols = np.arange(0,1,1/self.w)

        row_weights = rows * (1 - rows)
        col_weights = cols * (1 - cols)
        self.weights = row_weights[:, np.newaxis] * col_weights[np.newaxis, :]

        self.transformed_weights, _ = apply_proj(self.transform, self.weights)

        # self.inv = np.linalg.inv(self.transform)
        # self.is_rgb = len(image.shape) == 3


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

    full_domain = image_parts[0].domain
    for part in image_parts[1:]:
        full_domain = full_domain.union(part.domain)

    ext_images = [change_domain(part.transformed_image, part.domain, full_domain)
                   for part in image_parts]

    ext_weights = [change_domain(part.transformed_weights, part.domain, full_domain)[:,:,np.newaxis]    # always RGB?
                   for part in image_parts]
    sum_weights = sum(ext_weights)

    # we don't want to divide by zero
    sum_weights[sum_weights == 0] = 1

    full_image = sum(weight*img for weight, img in zip(ext_weights, ext_images))
    full_image /= sum_weights

    return full_image.astype(int)

