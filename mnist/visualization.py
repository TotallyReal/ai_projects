from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from typing import List, Tuple, Dict


def scatter_animation(data: List[Tuple[str, np.ndarray]], labels: np.ndarray):
    """
    Plots several animations of scatter plots with several colors each.

    'data' should be a list, where each item (name, frames) correspond to a different animation.

    frames : num_frames x 2 x n
    'labels' : 1 x n array with the labels for the data

    Change the save_animation_file parameter to save the result as a gif file save_animation_file.gif .
    """

    # Create the screens for the animation
    num_screens = len(data)
    fig, axes = plt.subplots(nrows=1 ,ncols=num_screens ,figsize=(4 *num_screens ,4))
    if num_screens == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    scatters = []

    num_frames = len(data[0][1])
    for ax, screen in zip(axes, data):
        name, pts_data = screen

        ax.set_title(name)
        ax.grid(True)

        pts_data = np.array(pts_data)
        scatters.append(ax.scatter(
            x = pts_data[0][0,:], y = pts_data[0][1,:],
            cmap = cmap, c = labels, s = 10
        ))
        max_x, max_y = np.array(pts_data).max(axis=2).max(axis=0)
        min_x, min_y = np.array(pts_data).min(axis=2).min(axis=0)
        ax.set_xlim(min_x *1.1, max_x *1.1)
        ax.set_ylim(min_y *1.1, max_y *1.1)

    frames_per_screen = [frm for _, frm in data]


    # Initialization function: plot the background of each frame
    def init():
        for scatter, frames in zip(scatters, frames_per_screen):
            scatter.set_offsets(frames[0].T)
        return scatters

    # Update function: update the data for each frame
    def update(frame_index):
        for scatter, frames in zip(scatters, frames_per_screen):
            scatter.set_offsets(frames[frame_index].T)
        return scatters

    # Create the animation
    animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    fig.legend(*scatters[0].legend_elements(), loc='upper right', title="Labels")

    # Display the animation
    plt.show()

    return animation


def plot_filters(arr :np.ndarray):
    """
    Plots up to 10 images in a grid.
    Input should be of shape k x n x n
    """
    k = len(arr)

    n_cols = 5
    for i in range(5):
        if k % i+1 == 0:
            n_cols = i+1
    n_rows = np.ceil(k/n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for img_arr, ax in zip(arr, axes):
        ax.imshow(img_arr, cmap='gray')
        ax.axis('off')
    plt.show()


def plot_more_filters2(
        weights0: np.ndarray, bias0: np.ndarray,
        inputs: np.ndarray, labels:np.ndarray,
        outputs_first: np.ndarray, outputs_last: np.ndarray):
    """
    Assumption: classifying 0/1 with a single hidden mid layer of size k

    weights0: k x res x res
    bias0:    k x 1
    weights1: 2 x k

    labels:   n x 1 over 0,1
    outputs0: n x k
    outputs1: n x 2
    """


    k = weights0.shape[0]
    grid_width = max(k, 10)
    grid_height = 5
    plt.figure(figsize=(grid_width * 2, grid_height * 2))

    filter_im = []

    for i, cur_weights in enumerate(weights0):
        ax = plt.subplot2grid((grid_height, grid_width), (0, i))
        filter_im.append(ax.imshow(cur_weights, cmap='gray'))
        ax.axis('off')
        ax.set_title(f'bias={bias0[i]:.3f}')

    input_ax = plt.subplot2grid((grid_height, grid_width), (2, grid_height-1), colspan = 2, rowspan=2)
    input_img = input_ax.imshow(inputs[0], cmap='gray')

    scatter_ax = plt.subplot2grid((grid_height, grid_width), (2, 0), colspan=grid_height - 2, rowspan=grid_height - 2)

    scatter = scatter_ax.scatter(x=outputs_last[:, 0], y=outputs_last[:, 1], c=labels, cmap=plt.get_cmap("tab10"), s = 5)

    def on_add(sel):
        result = outputs_first[sel.index]
        input_img.set_data(inputs[sel.index])
        for height, im in zip(result, filter_im):
            if height >= 0:
                im.set_cmap('gray')
            else:
                im.set_cmap('Reds')

    # mouse over event
    mplcursors.cursor(scatter, hover=True).connect("add", on_add)

    plt.show()


def plot_more_filters(
        weights0: np.ndarray, bias0: np.ndarray, weights1: np.ndarray,
        inputs: np.ndarray, labels:np.ndarray,
        outputs0: np.ndarray, outputs1: np.ndarray):
    """
    Assumption: classifying 0/1 with a single hidden mid layer of size k

    weights0: k x res x res
    bias0:    k x 1
    weights1: 2 x k

    labels:   n x 1 over 0,1
    outputs0: n x k
    outputs1: n x 2
    """

    k = weights0.shape[0]
    grid_width = max(k, 10)
    grid_height = 5
    plt.figure(figsize=(grid_width * 2, grid_height * 2))

    filter_im = []

    for i, cur_weights in enumerate(weights0):
        ax = plt.subplot2grid((grid_height, grid_width), (0, i))
        filter_im.append(ax.imshow(cur_weights, cmap='gray'))
        ax.axis('off')
        ax.set_title(f"{weights1[0][i]:.3f} : {weights1[1][i]:.3f}\nbias={bias0[i]:.3f}")

    bar_ax = plt.subplot2grid((grid_height, grid_width), (1, 0), colspan=k, rowspan=1)
    bar_ax.set_ylim(-5, 5)
    bar_ax.set_xlim(-0.5, k - 0.5)
    bars = bar_ax.bar(list(range(k)), [0] * k)

    input_ax = plt.subplot2grid((grid_height, grid_width), (2, grid_height-1), colspan = 2, rowspan=2)
    input_img = input_ax.imshow(inputs[0], cmap='gray')

    scatter_ax = plt.subplot2grid((grid_height, grid_width), (2, 0), colspan=grid_height - 2, rowspan=grid_height - 2)

    scatter = scatter_ax.scatter(x=outputs1[:, 0], y=outputs1[:, 1], c=labels, cmap=plt.get_cmap("tab10"))

    def on_add(sel):
        result = outputs0[sel.index]
        input_img.set_data(inputs[sel.index])
        for bar, height, im in zip(bars, result, filter_im):
            bar.set_height(height)
            if height >= 0:
                im.set_cmap('gray')
                bar.set_color('blue')
            else:
                im.set_cmap('Reds')
                bar.set_color('red')

    # mouse over event
    mplcursors.cursor(scatter, hover=True).connect("add", on_add)

    plt.show()
