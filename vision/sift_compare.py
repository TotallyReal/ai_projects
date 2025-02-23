import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing import List
from vision.sift import KeyPoint, process_image


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


image_name_1 = 'stitch5_10'
image_name_2 = 'stitch4_10'

compare_sift(image_name_1, image_name_2)

# image1 = mpimg.imread(f'images/{image_name_1}.jpg')
# image2 = mpimg.imread(f'images/{image_name_2}.jpg')
# compare_cv(image1, image2)

plt.show()