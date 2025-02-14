import functools
from abc import abstractmethod
import copy

import matplotlib.pyplot as plt

from misc import Timer

timer = Timer()
timer.time(print_time=False)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
from xp_data import XpData
from model_tester import test_model, train_model#, collect_train_progress
import numpy as np

from models import SimpleClassifier
from visualization import scatter_animation, plot_filters
import os


timer.time(msg='finish imports')


class MnistData:

    def __init__(self):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel
            transforms.ToTensor()  # Convert to tensor and normalize to [0, 1]
        ])

        # Load MNIST dataset
        self.train_dataset = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
        self.test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    @functools.lru_cache
    def restricted_data(self, up_to_label: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        res_train_dataset = Subset(
            self.train_dataset, [i for i, label in enumerate(self.train_dataset.targets) if label<up_to_label])
        res_test_dataset = Subset(
            self.test_dataset, [i for i, label in enumerate(self.test_dataset.targets) if label<up_to_label])
        return res_train_dataset, res_test_dataset

    @functools.lru_cache
    def train_loader(self, up_to_label:int, batch_size: int) -> DataLoader:
        train_dataset, _ = self.restricted_data(up_to_label)
        return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    @functools.lru_cache
    def test_loader(self, up_to_label:int, batch_size: int = -1) -> DataLoader:
        _, test_dataset = self.restricted_data(up_to_label)
        batch_size = batch_size if batch_size > 0 else len(test_dataset)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

@dataclass
class MnistParameters:
    seed: int = 1
    up_to_label: int = 10
    learning_rate: float = 0.001
    hidden: Tuple[int, ...] = field(default_factory=tuple)
    batch_size: int = 64
    epochs: int = 1

    def __post_init__(self):
        assert self.learning_rate > 0
        assert 2<=self.up_to_label<=10
        assert all([d>0 for d in self.hidden])
        assert self.batch_size > 0
        assert self.epochs > 0

params = MnistParameters(
    seed=2,
    up_to_label=2,
    learning_rate=0.001,
    hidden=(10,),
    batch_size=64,
    epochs=50)

# Prepare data
mnist_data = MnistData()

train_data, _ = mnist_data.restricted_data(up_to_label=params.up_to_label)
train_size = len(train_data)
train_loader = mnist_data.train_loader(up_to_label=params.up_to_label, batch_size=params.batch_size)
test_loader = mnist_data.test_loader(up_to_label=params.up_to_label)

# Prepare model
torch.manual_seed(params.seed)
model = SimpleClassifier(dimensions=[28 * 28] + list(params.hidden) + [params.up_to_label])


class OutputCollector:
    """
    Train the model, and after each training batch step run it on the test data, and collect the output
    of any linear layer which has a 2-dimensional output.

    Returns a dictionary progress where
        progress[layer][label] = list[ 2 x n arrays ]
    Where the layer is a nn.Linear and not just a number, label is the label in [0,9],
    the list elements correspond to the learning steps, and each element is a 2 x n array with the output
    of that step.
    """

    def __init__(self, model: SimpleClassifier, test_loader: DataLoader, up_to_label: int):
        self.model = model
        self.test_loader = test_loader
        self.up_to_label = up_to_label

        # Make sure the linear layers retain their output
        self.outputs = dict()

        self.progress_output = dict()
        for linear_layer in model.linear_layers():
            linear_layer.register_forward_hook(self._save_output)
            self.progress_output[linear_layer] = {label: [] for label in range(self.up_to_label)}

    def _save_output(self, model, input, output):
        self.outputs[model] = output

    # After each learning step, run the model on the test data and collect the output
    def progress_callback(self, epoch: int, num_data_points: int):
        self.model.eval()
        # collect linear layers output on the test dataset (one batch)
        with torch.no_grad():
            test_data_input, test_data_output = next(iter(self.test_loader))
            X_test = test_data_input #.to(device)
            self.model(X_test)
            for layer, progress in self.outputs.items():
                for label in range(self.up_to_label):
                    self.progress_output[layer][label].append(progress[test_data_output==label].T)
        self.model.train()

    def generate_animation(self, save_animation_file: str = ''):
        frames_data = [
            (f'layer {i}', self.progress_output[linear_layer])
            for i, linear_layer in enumerate(self.model.linear_layers())
            if linear_layer.out_features == 2
        ]
        scatter_animation(frames_data, save_animation_file=save_animation_file)


# <editor-fold desc=" ------------------------ Training with animation on 2D outputs ------------------------">

# collector = OutputCollector(model=model, test_loader=test_loader, up_to_label=params.up_to_label)
#
# train_model(model=model, data_loader=train_loader, train_size=train_size,
#             epochs=params.epochs, learning_rate=params.learning_rate,
#             train_step_callback=collector.progress_callback)
#
# hidden_str = '_'.join(str(d) for d in params.hidden)
# collector.generate_animation(save_animation_file=f'media/labels_{params.up_to_label}_hidden_{hidden_str}')

# </editor-fold>

# <editor-fold desc=" ------------------------ Plot filters from first layer ------------------------">

def plot_first_layer_images(epoch: int, num_data_points: int):
    if num_data_points>0:
        return
    weights = model.get_linear_layer(1).weight.detach().cpu().numpy()
    arr = model.get_linear_layer(0).weight.detach().cpu().numpy()
    arr = arr.reshape(arr.shape[0], 28, 28)
    # plot_filters(arr)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5 * 2, 2 * 2))
    fig.suptitle(f'Epoch {epoch}')
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(arr[i], cmap='gray')
        ax.set_title(f"{weights[0][i]:.3f} ; {weights[1][i]:.3f}", fontsize=12)
        ax.axis('off')
    save_path = f'images/seed{params.seed}/filters_{epoch}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

train_model(model=model, data_loader=train_loader, train_size=train_size,
            epochs=params.epochs, learning_rate=params.learning_rate,
            train_step_callback=plot_first_layer_images)

# </editor-fold>

# <editor-fold desc=" ------------------------ Compute error rates ------------------------">

print('Train data:')
train_error_rate = test_model(model, train_loader)
print('Test data:')
test_error_rate = test_model(model, test_loader)

# </editor-fold>

exit(0)

def run_hyper_parameters(
        params: MnistParameters,
        hidden_layers_param: Optional[List[Tuple[int]]] = None ,
        batch_sizes: Optional[List[int]] = None,
        learning_rates : Optional[List[float]] = None):

    if hidden_layers_param is None:
        hidden_layers_param = [params.hidden]
    if batch_sizes is None:
        batch_sizes = [params.batch_size]
    if learning_rates is None:
        learning_rates = [params.learning_rate]

    params = copy.deepcopy(params)

    for hidden_layers in hidden_layers_param:
        params.hidden = hidden_layers

        # Create model
        torch.manual_seed(params.seed)
        model = SimpleClassifier([28 * 28] + list(params.hidden) + [params.up_to_label])
        initial_state = model.state_dict()


        for batch_size in batch_sizes:
            params.batch_size = batch_size
            train_loader = mnist_data.train_loader(up_to_label=params.up_to_label, batch_size=params.batch_size)

            for learning_rate in learning_rates:
                params.learning_rate = learning_rate

                model.load_state_dict(initial_state)

                yield model, params, train_loader

running_data = XpData(
    file_path=f'running_data{params.up_to_label}.csv',
    data_cls=MnistParameters,
    values = dict(train_error_rate=float, test_error_rate=float))
saved_size = len(running_data)


initial_params = MnistParameters(
    seed=1,
    up_to_label=2,
    learning_rate=1,
    hidden=(),
    batch_size=1,
    epochs=1)

save_data = True

# To run over different epochs, use the callback
for model, params, train_loader in run_hyper_parameters(
        params = initial_params,
        hidden_layers_param = [(), (2,)],
        batch_sizes = [64, 128],
        learning_rates = [0.001]
    ):

    print('----------------------------------------------------')
    if running_data.contains_index(params):
        print(f'Skipping {params}')
        continue
    print(f'Running for parameters {params=}')

    train_model(
        model=model, data_loader=train_loader, train_size=train_size,
        epochs=params.epochs, learning_rate=params.learning_rate)

    print('Train data:')
    train_error_rate = test_model(model, train_loader)
    print('Test data:')
    test_error_rate = test_model(model, test_loader)

    running_data.add_entry(
        params,
        train_error_rate=train_error_rate, test_error_rate=test_error_rate)
    if save_data and len(running_data) >= saved_size + 20:
        running_data.save()

if save_data:
    running_data.save()


