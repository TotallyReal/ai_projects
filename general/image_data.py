from dataclasses import dataclass
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Optional

from misc import Timer

timer = Timer()
timer.time(print_time=False)

import torch
from torch.utils.data import DataLoader, Subset, Dataset

from mnist import visualization
from general.models import SimpleClassifier


timer.time(msg='finish imports')

@dataclass
class ImageDataParameters:
    up_to_label: int = 10
    batch_size: int = 64
    train_size: int = -1
    test_size: int = -1

    def __post_init__(self):
        assert 2<=self.up_to_label<=10
        assert self.batch_size > 0


class ImageData:

    def __init__(self,
                 train_dataset: Dataset, test_dataset: Dataset,
                 label_names: List[str] ):

        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset

        self.label_count = len(label_names)
        self.label_names = label_names
        self.name_to_label = {name: i for i, name in enumerate(self.label_names)}

    def to_labels(self, label_names: List[str]) -> List[int]:
        return [self.name_to_label[name] for name in label_names]

    @functools.lru_cache
    def restricted_data(self, up_to_label: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        res_train_dataset = Subset(
            self.train_dataset, [i for i, label in enumerate(self.train_dataset.targets) if label<up_to_label])
        res_test_dataset = Subset(
            self.test_dataset, [i for i, label in enumerate(self.test_dataset.targets) if label<up_to_label])
        return res_train_dataset, res_test_dataset

    def plot_examples(self, labels: Optional[List[int]] = None):
        if labels is None:
            labels = list(range(self.label_count))

        position = {label: i for i,label in enumerate(labels)}
        images = [None] * len(labels)
        counter = len(labels)

        for image, label in self.train_dataset:
            if counter == 0:
                break
            if images[position[label]] is None:
                # squeezing the gray scale channel which is of dimension 1
                images[position[label]] = image.squeeze()
                counter -= 1
        visualization.plot_images(images, labels=[self.label_names[i] for i in labels])

    @functools.lru_cache
    def train_loader(self, up_to_label:int, batch_size: int) -> DataLoader:
        train_dataset, _ = self.restricted_data(up_to_label)
        return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    def train_loaders_from_parameters(self, up_to_label: List[int], batch_sizes: List[int]):
        for label_bound in up_to_label:
            train_data, test_data = self.restricted_data(up_to_label=label_bound)
            train_size = len(train_data)
            test_size = len(test_data)
            for batch_size in batch_sizes:
                yield (self.train_loader(label_bound, batch_size),
                       ImageDataParameters(label_bound, batch_size, train_size, test_size))

    @functools.lru_cache
    def test_loader(self, up_to_label:int, batch_size: int = -1) -> DataLoader:
        _, test_dataset = self.restricted_data(up_to_label)
        batch_size = batch_size if batch_size > 0 else len(test_dataset)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



def plot_first_layer_images(model: SimpleClassifier, title:str, save_path: str = ''):
    """
    Plots the filters of the first layer after each epoch and show them
    Change 'save_path' to a file path in order to save the output (instead of showing it in plt).
    """
    special_case = False
    weights = None
    if model.number_of_linear_layers() == 2:
        weights = model.get_linear_layer(1).weight.detach().cpu().numpy()
        special_case = (weights.shape[0] == 2)
    arr = model.get_linear_layer(0).weight.detach().cpu().numpy()
    arr = arr.reshape(arr.shape[0], 28, 28)
    bias = model.get_linear_layer(0).bias.detach().cpu().numpy()
    # plot_filters(arr)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5 * 2, 2 * 2 + 1))
    fig.suptitle(title)
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    if special_case:
        for ax, img_arr, b, w in zip(axes, arr, bias, weights.T):
            ax.imshow(img_arr, cmap='gray')
            ax.set_title(f'{w[0]:.3f} ; {w[1]:.3f}\nBias={b:.3f}', fontsize=12)
    else:
        for i, (ax, img_arr, b) in enumerate(zip(axes, arr, bias)):
            ax.imshow(img_arr, cmap='gray')
            ax.set_title(f'Bias={b:.3f}', fontsize=12)
    if save_path != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# <editor-fold desc=" ------------------------ Training with animation on 2D outputs ------------------------">

class OutputProgressCollector:
    """
    Train the model, and after each training batch step run it on the test data, and collect the output
    of any linear layer which has a 2-dimensional output.

    Can be used to generate an animation.
    """

    def __init__(self, model: SimpleClassifier, test_input: torch.Tensor, test_output: torch.Tensor):
        self.model = model
        self.test_input = test_input
        self.test_output = test_output.detach().numpy()

        # Make sure the linear layers retain their output
        self.outputs = model.collect_output()

        self.progress_output = dict()
        for linear_layer in model.linear_layers():
            self.progress_output[linear_layer] = []

    # After each learning step, run the model on the test data and collect the output
    def progress_callback(self, epoch: int, num_data_points: int):
        self.model.eval()
        # collect linear layers output on the test dataset (one batch)
        with torch.no_grad():
            self.model(self.test_input)
            for layer, progress in self.outputs.items():
                self.progress_output[layer].append(progress.T)
        self.model.train()

    def generate_animation(self, save_animation_file: str = ''):
        # (str,  num_frames x 2 x batch_size)
        frames_data = [
            (f'layer {i}', np.stack(self.progress_output[linear_layer], axis=0))
            for i, linear_layer in enumerate(self.model.linear_layers())
            if linear_layer.out_features == 2
        ]
        animation = visualization.scatter_animation(frames_data, labels=self.test_output)

        if save_animation_file != '':
            animation.save(f"{save_animation_file}.gif", fps=10, writer="imagemagick")  # or use .mp4 for video format

# </editor-fold>

