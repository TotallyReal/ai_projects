import cv2
from vision import reconstruction3d, sift_compare
import numpy as np
import pytest

EPSILON = 0.01

@pytest.fixture(autouse=True)
def set_random_seed():
    seed = np.random.randint(0, 1000000000)
    print(f'Seed is {seed}')
    np.random.seed(seed)

def to_pos_times(position: np.ndarray):
    return np.array([
            [     0      , -position[2], position[1]],
            [ position[2],      0      ,-position[0]],
            [-position[1],  position[0],     0      ]
    ])

class RelImage:

    def __init__(self, image: np.ndarray, position: np.ndarray, rotation: np.ndarray, calibration: np.ndarray):
        self.image = image
        self.position = position
        self.pos_times = to_pos_times(position)
        self.rotation = rotation
        self.calibration = calibration
        if len(self.image.shape) == 3:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            self.gray_image = self.image

        self.keypoints = None
        self.descriptos = None
        self.calib_ing = np.linalg.inv(calibration)

        self.fundamental = self.calib_ing.T @ self.pos_times @ rotation @ self.calib_ing
        self.fundamental /= np.linalg.norm(self.fundamental)


def test_unity_images():

    return
    # TODO: reopen this test ...
    np.set_printoptions(precision=3)
    print('\n')
    calibration_matrix = reconstruction3d.calibrate_from_images(
        reconstruction3d.load_images_from('images/calibration', 'png'), print_matrix=False)

    position_names = [
        'center', 'forward', 'right', 'up', 'look_right','look_up', 'forward_look_right'
    ]

    rel_images = dict(
        center = RelImage(
            image = cv2.imread(f'images/center.png', cv2.IMREAD_COLOR_RGB),
            position = np.array([0,0,0]),
            rotation = np.eye(3),
            calibration = calibration_matrix
        ),
        right = RelImage(
            image = cv2.imread(f'images/right.png', cv2.IMREAD_COLOR_RGB),
            position = np.array([1,0,0]),
            rotation = np.eye(3),
            calibration = calibration_matrix
        ),
    )

    image_system = sift_compare.ImageSystem([rel_images['center'].image])


    for key, rel_image in rel_images.items():
        if key == 'center':
            continue
        index = image_system.add_image(rel_image.image)
        matching = image_system.matching(from_index=0, to_index=index, threshold=0.5)

        fundamental = reconstruction3d.least_median_fundamental(matching)
        assert (np.allclose(rel_image.fundamental, fundamental, atol=EPSILON) or
                np.allclose(rel_image.fundamental, -fundamental, atol=EPSILON))

        rel_pos, rel_rot = reconstruction3d.find_relative_position(fundamental, calibration_matrix, matching)
        assert np.allclose(rel_image.position, rel_pos, atol=EPSILON)
        assert np.allclose(rel_image.rotation, rel_rot, atol=EPSILON)

def print_transform_steps(P: np.ndarray, K: np.ndarray, C: np.ndarray):
    np.set_printoptions(precision=3)
    print(f'Parameters: ')
    print(f'{P=}')
    print(f'{K=}')
    print(f'{C=}')

    Ptimes = np.array([
        [0,-P[2,0],P[1,0]],
        [P[2,0],0,-P[0,0]],
        [-P[1,0],P[0,0],0]
    ])
    print(f'\nFundamental + Essential matrix: ')
    E = Ptimes @ K
    Cinv = np.linalg.inv(C)
    F = Cinv.T @ E @ Cinv
    print(f'{F=}')
    print(f'{E=}')
    print(f'\nSVD: ')
    U, S, Vh = np.linalg.svd(E)
    print(f'{U=}')
    print(f'{S=}')
    print(f'{Vh=}')
    print('-----------------------\n')

def generate_points(P: np.ndarray, K: np.ndarray, C:np.ndarray, num_points: int):# Generates points in 3D, and relative to the cameras
    num_points = 30
    np.random.seed(0)
    points3d = np.random.randint(-50, 50, size=(num_points, 3)) + np.array([0,0,100])
    # points3d = np.mgrid[-5:6, -5:6, 5:16].reshape(3, -1).T
    # print(points3d)

    points3d_ori = points3d @ C.T
    points3d_rel = (points3d - P) @ K @ C.T

    # Keep only points which are not at infinity relative to one of the cameras
    finite_positions = (points3d_ori[:, 2] != 0) & (points3d_rel[:, 2] != 0)
    points3d = points3d[finite_positions]
    points3d_ori = points3d_ori[finite_positions]
    points3d_rel = points3d_rel[finite_positions]

    points2d_ori = points3d_ori[:, :2] / points3d_ori[:, 2][:,np.newaxis]
    points2d_rel = points3d_rel[:, :2] / points3d_rel[:, 2][:,np.newaxis]
    return points3d, points2d_ori, points2d_rel

@pytest.fixture(params=[
    np.array([5,0,0]),
    # np.array([0,5,0]), np.array([0,0,5]), np.array([1,2,3])
])
def position(request) -> np.ndarray:
    return request.param

@pytest.fixture(params=[
    # np.eye(3),
    np.array([[1,1,1],[1,0,2],[0,-1,3]])
])
def calibration_mat(request) -> np.ndarray:
    return request.param

@pytest.fixture(params=[
    # np.eye(3),
    np.array([[1,1,1],[1,0,2],[0,-1,3]])
])
def rotation(request) -> np.ndarray:
    rotation, _ = np.linalg.qr(request.param) # Just to help me a bit
    return rotation

@pytest.fixture
def fundamental(position: np.ndarray, rotation: np.ndarray, calibration_mat: np.ndarray):
    c_inv = np.linalg.inv(calibration_mat)
    fundamental = c_inv.T @ to_pos_times(position) @ rotation @ c_inv
    fundamental /= np.linalg.norm(fundamental)
    return fundamental

def fundamental_coords(fundamental: np.ndarray, matching:np.ndarray):
    arr = np.array([[u1 * v1, u1 * v2, u1, u2 * v1, u2 * v2, u2, v1, v2, 1]
                    for (u1, u2), (v1, v2) in matching])

    U, S, Vt = np.linalg.svd(arr)
    return Vt @ fundamental.reshape(9,1)


def generate_matching(
        num_points: int, position: np.ndarray, rotation: np.ndarray, calibration_mat: np.ndarray, add_noise:bool) -> np.ndarray:

    points3d, points2d_ori, points2d_rel = generate_points(position, rotation, calibration_mat, num_points)

    if add_noise:
        # TODO: Understand how noise affect the computation of the fundamental matrix
        #       For now test without noise
        np.random.seed(0)
        scale = np.max(np.max(points2d_ori, axis=0) - np.min(points2d_ori, axis=0))
        points2d_ori += np.random.normal(0, scale/200, points2d_ori.shape)
        # points2d_rel += np.random.normal(0, 0.01, points2d_rel.shape)

    return np.stack([points2d_ori, points2d_rel], axis=1)


@pytest.mark.parametrize('add_noise', [False])
def test_fundamental(
        position: np.ndarray, rotation: np.ndarray, calibration_mat: np.ndarray,
        fundamental: np.ndarray, add_noise: bool):

    num_points = 10
    matching = generate_matching(num_points, position, rotation, calibration_mat, add_noise)

    res_fundamental = reconstruction3d.least_median_fundamental(matching)

    assert abs(np.linalg.norm(fundamental)-1) < EPSILON
    assert abs(np.linalg.norm(res_fundamental)-1) < EPSILON

    distances = reconstruction3d.epipolar_distances(matching, res_fundamental)
    assert np.median(distances) < EPSILON
    if not add_noise:
        # without any noise, we should expect the max to be small as well
        assert np.max(distances) < EPSILON

    assert (np.allclose(fundamental, +res_fundamental, atol=EPSILON) or
            np.allclose(fundamental, -res_fundamental, atol=EPSILON)), \
        f'Bad fundamental: \n{fundamental}\n > - < \n {res_fundamental}'

@pytest.mark.parametrize('add_noise', [False])
def test_decomposition(
        position: np.ndarray, rotation: np.ndarray, calibration_mat: np.ndarray,
        fundamental: np.ndarray, add_noise: bool):
    print('\n')

    num_points = 10
    matching = generate_matching(num_points, position, rotation, calibration_mat, add_noise)

    res_fundamental = reconstruction3d.least_median_fundamental(matching)
    rel_pos, rel_rot = reconstruction3d.find_relative_position(res_fundamental, calibration_mat, matching)

    # Normalize parameters
    position = position.astype(float)
    position/=np.linalg.norm(position)
    rel_pos /=np.linalg.norm(rel_pos)

    assert np.linalg.norm(position-rel_pos)<0.15, f'Bad position:\nReal position: {position}\nComputed position: {rel_pos}'
    assert np.allclose(rotation, rel_rot, atol=0.01), f'Bad rotation:\nReal rotation: \n{position}\nComputed rotation: \n{rel_pos}'

def test_triangulation(position: np.ndarray, rotation: np.ndarray, calibration_mat: np.ndarray):
    num_points = 30
    points3d, points2d_ori, points2d_rel = generate_points(position, rotation, calibration_mat, num_points)

    tri_points = np.array([reconstruction3d.triangulate(ori2d, rel2d,calibration_mat, rotation, position)
                  for ori2d, rel2d in zip(points2d_ori, points2d_rel)])

    print(np.allclose(tri_points, points3d, atol=EPSILON))

