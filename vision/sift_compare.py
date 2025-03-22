import cv2
from functools import cache
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing import List, Tuple
from vision.sift import KeyPoint, process_image

Point = Tuple[float, float]  # for (x,y) points


class ImageSystem:

    @staticmethod
    def point_matching(
            matcher: cv2.DescriptorMatcher,
            keypoints1: List[cv2.KeyPoint], descriptors1: List[np.ndarray],
            keypoints2: List[cv2.KeyPoint], descriptors2: List[np.ndarray],
            threshold: float = 0.5) -> Tuple[List[Tuple[Point, Point]], List[cv2.DMatch]]:
        """
        Returns an array n x 2 x 2 of good matches, namely arr[i][0] -> arr[i][1] is a match.
        """
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches: List[cv2.DMatch] = []
        for best_match, second_best in matches:
            if best_match.distance < threshold * second_best.distance:
                good_matches.append(best_match)

        # Create system of linear equations:
        return np.array([(keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt) for match in good_matches]), good_matches

    def __init__(self, images: List[np.ndarray]):
        self.is_rgb = len(images[0].shape) == 3
        self.images = images
        self.gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images] if self.is_rgb else images

        # Generate SIFT keypoints and descriptors
        self.sift = cv2.SIFT_create()
        self.sift_data = [self.sift.detectAndCompute(img, None) for img in self.gray_images]

        # Matcher
        self.matcher = cv2.BFMatcher()

        # Another possible matcher:
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def add_image(self, image: np.ndarray) -> int:
        self.images.append(image)
        self.gray_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if self.is_rgb else image)
        self.sift_data.append(self.sift.detectAndCompute(self.gray_images[-1], None))
        return len(self.images) - 1


    @cache
    def matching(self, from_index: int, to_index: int, threshold: float = 0.7) -> List[Tuple[Point, Point]]:
        return ImageSystem.point_matching(
            self.matcher, *self.sift_data[from_index], *self.sift_data[to_index], threshold=threshold)[0]

    @cache
    def cv_matchings(self, from_index: int, to_index: int, threshold: float = 0.7) -> List[cv2.DMatch]:
        return ImageSystem.point_matching(
            self.matcher, *self.sift_data[from_index], *self.sift_data[to_index], threshold=threshold)[1]

    def matching_image(self, from_index: int, to_index: int, threshold: float = 0.7):
        return cv2.drawMatches(
            self.images[from_index], self.keypoints(from_index),
            self.images[to_index],   self.keypoints(to_index),
            self.cv_matchings(from_index, to_index, threshold),
            outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def keypoints(self, index:int):
        return self.sift_data[index][0]

def plot_comparisons(image1, points1, descriptors1, image2, points2, descriptors2):
    diff = descriptors1[:, None, :] - descriptors2[None, :, :]
    distances = np.sum(diff**2, axis=2)
    sorted_distances = sorted(distances.reshape(-1))

    indices = np.where(distances <= sorted_distances[min(50, len(sorted_distances) - 1)])  # 2 x n
    print(f'found {len(indices[0])} comparisons')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    from matplotlib.patches import ConnectionPatch

    # Display the images
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    for index1, index2 in zip(indices[0], indices[1]):
        x1, y1 = points1[index1]
        x2, y2 = points2[index2]
        con = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y2), coordsA="data", coordsB="data",
                              axesA=ax[0], axesB=ax[1], color=np.random.rand(3, ))
        ax[1].add_artist(con)

def generate_image_data(image_name: str):
    data_file_path = f'data/{image_name}.jpg'
    if os.path.exists(data_file_path):
        with open(data_file_path, 'rb') as f:
            keypoints: List[KeyPoint] = pickle.load(f)
        descriptors = np.array([kpt.hist for kpt in keypoints])
        keypoints = [kp.position for kp in keypoints if len(kp.hist) > 1]
        return keypoints, descriptors
    else:
        image = mpimg.imread(f'images/{image_name}.jpg')
        keypoints, descriptors, _ = process_image(np.mean(image,axis=2))
        with open(data_file_path, 'wb') as f:
            pickle.dump(keypoints, f)
        return keypoints, descriptors

def compare_sift(image_name_1: str, image_name_2: str):
    keypoints1, descriptors1 = generate_image_data(image_name_1)
    print(f'Key points 1 length = {len(keypoints1)}')
    keypoints2, descriptors2 = generate_image_data(image_name_2)
    print(f'Key points 2 length = {len(keypoints2)}')

    image1 = mpimg.imread(f'images/{image_name_1}.jpg')
    image2 = mpimg.imread(f'images/{image_name_2}.jpg')
    plot_comparisons(image1, keypoints1, descriptors1, image2, keypoints2, descriptors2)
    plt.title('SIFT comparison')

def compare_cv(image1: np.ndarray, image2: np.ndarray, threshold: float = 0.7):
    """
    Standard matching between two images using the open cv library.
    'threshold' should be a nonnegative number indicating how good of a matching we are looking for.
    threshold==0 means that we only allow perfect matching (which basically only possible if you match a picture
    with itself, and hope that no noise somehow got into the computations).
    """

    image_system = ImageSystem([image1, image2])

    matched_img = image_system.matching_image(0,1, threshold=threshold)
    plt.imshow(matched_img)
    plt.title("SIFT Matches")
    plt.axis('off')
    plt.show()

    # cv2.imshow("SIFT Matches", matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



# image_name_1 = 'stitch5_10'
# image_name_2 = 'stitch4_10'
#
# compare_sift(image_name_1, image_name_2)
# plt.show()

# image1 = mpimg.imread(f'images/{image_name_1}.jpg')
# image2 = mpimg.imread(f'images/{image_name_2}.jpg')
# compare_cv(image1, image2)

# plt.show()