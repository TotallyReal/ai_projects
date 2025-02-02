import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple, Dict
from matplotlib.patches import Rectangle

def scatter_animation(data: List[Tuple[str, Dict[str, List[np.ndarray]]]], save_animation_file: str = ''):
    """
    Plots several animations of scatter plots with several colors each.

    Input should be a list, where each item (name, dictionary) correspond to a different animation.
    This dictionary is from labels to list of frames, where each frame are the 2D points needed
    to plot at that frame, which should be in an array 2 x n.

    All animations should have the same amount of frames, and the same labels.

    Change the save_animation_file parameter to save the result as a gif file save_animation_file.gif .
    """

    # Create the screens for the animation
    num_screens = len(data)
    fig, axes = plt.subplots(nrows=1 ,ncols=num_screens ,figsize=( 4 *num_screens ,4))
    if num_screens == 1:
        axes = [axes]

    all_frames = []
    all_plots = []
    colors = plt.get_cmap("tab10").colors

    num_frames = len(list(data[0][1].values())[0]) # ?!??@?#!?@#?!@?#!@?#!!!!
    for ax, screen in zip(axes, data):
        name, pts_data = screen

        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.legend()
        ax.grid(True)

        # Prepare the different scatters in each screen
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for label, frames in pts_data.items():
            assert len(frames) == num_frames
            all_plots.append(ax.plot([], [], 'o', color = colors[label], label=label, markersize=1.5)[0])
            all_frames.append(frames)

            # Make sure the region in each screen contains all the points in all the frames
            cur_max_x, cur_max_y = np.array(frames).max(axis=2).max(axis=0)
            cur_min_x, cur_min_y = np.array(frames).min(axis=2).min(axis=0)
            max_x = max(max_x, cur_max_x)
            max_y = max(max_y, cur_max_y)
            min_x = min(min_x, cur_min_x)
            min_y = min(min_y, cur_min_y)

        ax.set_xlim(min_x *1.1, max_x *1.1)
        ax.set_ylim(min_y *1.1, max_y *1.1)

    # Initialization function: plot the background of each frame
    def init():
        for plot in all_plots:
            plot.set_data([], [])
        return all_plots

    # Update function: update the data for each frame
    def update(frame):
        for plot, frames in zip(all_plots, all_frames):
            plot.set_data(frames[frame][0], frames[frame][1])
        return all_plots

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    fig.legend(labels=list(data[0][1].keys()), loc='upper right', title="Labels")
    if save_animation_file != '':
        ani.save(f"{save_animation_file}.gif", fps=10, writer="imagemagick")  # or use .mp4 for video format

    # Display the animation
    plt.show()


def plot_filters(arr :np.ndarray):
    """
    Plots up to 10 images in a grid.
    Input should be of shape num_labels x n x n
    """
    num_labels = len(arr)

    if num_labels <= 5:
        n_rows = 1
        n_cols = num_labels
    else:
        n_rows = 2
        n_cols = int(np.ceil(num_labels / 2))
    if num_labels == 9:
        n_rows = 3
        n_cols = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(arr[i], cmap='gray')
        ax.axis('off')
    plt.show()