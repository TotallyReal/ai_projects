import matplotlib.pyplot as plt
import numpy as np
from colorsys import hsv_to_rgb
import open3d as o3d
import pyvista as pv
from scipy.spatial import cKDTree, Delaunay



def generate_sphere_points_and_colors(radius=1, num_points=1000):
    # Generate spherical coordinates
    phi = np.linspace(0, np.pi, num_points)  # Polar angle
    theta = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle

    # Create meshgrid for spherical coordinates
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Flatten arrays to create a list of 3D points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Calculate hue based on azimuthal angle (theta)
    hue = (np.arctan2(y.flatten(), x.flatten()) + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Calculate saturation based on polar angle (phi)
    saturation = np.sin(phi.flatten())  # Saturation is high at the equator, low at poles

    # Brightness is set to 1 for all points
    brightness = np.ones_like(hue)

    # Convert HSB to RGB
    rgb_colors = np.array([hsv_to_rgb(h, s, v) for h, s, v in zip(hue, saturation, brightness)])
    rgb_colors = np.clip(rgb_colors, 0, 1)  # Ensure colors are within [0, 1]

    return points, rgb_colors


def visualize_with_pyplot(points, colors):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud with colors
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)

    # Add color bar to show the mapping of colors
    fig.colorbar(scatter)

    # Show plot
    plt.show()


def visualize_with_open_3d(points, colors):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def prepare_point_cloud(points, colors):
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=2)  # k=2 because the nearest neighbor is the point itself
    nearest_distances = distances[:, 1]
    max_dist = np.percentile(nearest_distances, 70)

    mask = nearest_distances < max_dist
    points = points[mask]
    colors = colors[mask]
    nearest_distances = nearest_distances[mask]

    mid_dist = np.percentile(nearest_distances, 70)
    print(f'{mid_dist=}')

    # plt.hist(nearest_distances,bins=100)
    # plt.show()

    neighbors = tree.query(points, k=3, distance_upper_bound=mid_dist)
    neighbor_dist1 = neighbors[0][:, 1]
    neighbor_index1 = neighbors[1][:, 1]
    neighbor_dist2 = neighbors[0][:, 2]
    neighbor_index2 = neighbors[1][:, 2]

    valid_edges = [(2, i, j) for i, j, dist in zip(range(len(points)), neighbor_index1, neighbor_dist1) if
                   dist < mid_dist and i < j]
    edges = np.array(valid_edges)
    valid_faces = [(3, i, j, k) for i,j,k,dist in zip(range(len(points)), neighbor_index1, neighbor_index2, neighbor_dist2)
                   if dist < mid_dist]
    faces = np.array(valid_faces)


    # neighbors = tree.query(points, k=5, distance_upper_bound=mid_dist)
    # indices = np.arange(len(points))
    # mask = neighbors[0][:, 4] != float('inf')
    # indices = indices[mask]
    # neighbor_dist = neighbors[0][mask, :]
    # neighbor_idx = neighbors[1][mask, :]
    # neighbor_pts = points[neighbor_idx] - points[indices][:, np.newaxis, :]

    return points, colors, edges, faces


def visualize_with_pyvista(points, colors, alpha=0.05):
    points, colors, edges, faces = prepare_point_cloud(points, colors)
    rgba = np.hstack([colors, np.ones((len(colors),1))])

    point_cloud_vista = pv.PolyData(points)
    point_cloud_vista.point_data['Colors'] = rgba

    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud_vista, scalars='Colors', rgba=True, point_size=5)

    # Define functions for rotating the camera
    def rotate_clockwise():
        plotter.camera.Roll(-5)
        plotter.render()

    def rotate_counterclockwise():
        plotter.camera.Roll(5)
        plotter.render()

    # Bind arrow keys
    plotter.add_key_event("Right", rotate_clockwise)
    plotter.add_key_event("Left", rotate_counterclockwise)

    plotter.camera.position = (0, 0, -10)
    plotter.camera.up = (0, -1, 0)

    plotter.show()

    # plotter.open_gif("wave.gif")
    #
    # num_frames = 60
    # for t in np.linspace(0,1,num_frames):
    #     angle = np.sin(2*np.pi * t)*np.pi/3
    #     plotter.camera.position = (8*np.sin(angle), 0, -8*np.cos(angle))
    #     plotter.write_frame()
    #
    # plotter.close()


def visualize_surface_with_pyvista(points, colors, alpha=0.05):
    points, colors, edges, faces = prepare_point_cloud(points, colors)
    rgba = np.hstack([colors, np.ones((len(colors),1))])

    # lines = np.hstack(edges),
    # faces = np.hstack(faces)
    point_cloud_vista = pv.PolyData(
        points,
    )
    point_cloud_vista.point_data['Colors'] = rgba
    volume = point_cloud_vista.delaunay_3d(alpha=alpha)
    shell = volume.extract_geometry()

    plotter = pv.Plotter()
    plotter.add_mesh(shell, rgba=True)
    plotter.camera.position = (0, 0, -10)
    plotter.camera.up = (0, -1, 0)
    plotter.show()




# points , colors = generate_sphere_points_and_colors(1, 30)
# visualize_with_pyvista(points, colors)

