import numpy as np
from typing import Callable, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import functools


def auto_plot(plotter: Callable) -> Callable:

    @functools.wraps(plotter)
    def wrapper(*args, **kwargs):
        if (len(args) > 0 and args[0] is not None) or ('ax' in kwargs and kwargs['ax'] is not None):
            return plotter(*args, **kwargs)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        if 'ax' in kwargs:
            kwargs['ax'] = ax
        else:
            args = (ax,) + args[1:]

        result = plotter(*args, **kwargs)

        plt.tight_layout()
        plt.show()

        return result

    return wrapper

@auto_plot
def view_pair_probabilities(
        ax: Axes,
        alphabet: List[str],
        probabilities: np.ndarray,
        title: str = 'Probabilities for letter pairs'):
    abc_size = len(alphabet)

    sns.heatmap(
        probabilities, fmt="", cmap="viridis", cbar=True,
        square=True, linewidths=0.5, ax=ax)

    # Label axes
    ax.set_xticks(np.arange(abc_size) + 0.5)
    ax.set_yticks(np.arange(abc_size) + 0.5)
    ax.set_xticklabels(labels=alphabet)
    ax.set_yticklabels(labels=alphabet)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Title and layout
    ax.set_title(title)

@auto_plot
def show_letter_2_embedding(
        ax: Axes, alphabet: List[str], weights:np.ndarray, title:str = 'letter embedding'):

    ax.set_title(title)
    if weights.shape[0] != 2:
        return
    ax.scatter(weights[0, :].data, weights[1, :].data, s = 300, color="black")
    for i in range(len(alphabet)):
        ax.text(weights[0, i].item(), weights[1, i].item(), alphabet[i],
                 ha="center", va="center", color="white", fontsize = 15)
    ax.grid('minor')