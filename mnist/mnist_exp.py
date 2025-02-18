from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Optional, Iterator, Callable, TypeVar

from misc import Timer

timer = Timer()
timer.time(print_time=False)

from torchvision import datasets, transforms
import torch

from mnist import visualization
from general.model_tester import test_model, train_model, TrainingParameters
from general.models import SimpleClassifier, ModelParameters
from general.xp_data import XpData

from general.image_data import ImageData, ImageDataParameters
from general import analyse_data, image_data


timer.time(msg='finish imports')


class MnistData(ImageData):

    def __init__(self, data_root_dir: str = ''):
        """
        Leave the data_root_dir empty to load the data (if needed) into the directory
        of this file.
        Otherwise, it loads into data_root_dir.
        For example, you can use data_root_dir = './data' to load it into the subdirectory
        data inside the directory of the file from which the script is running.
        """
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel
            transforms.ToTensor()  # Convert to tensor and normalize to [0, 1]
        ])

        # Load MNIST dataset
        if data_root_dir == '':
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.join(script_dir, "data")

        train_dataset = datasets.MNIST(root=root_dir, train=True,  transform=transform, download=True)
        test_dataset  = datasets.MNIST(root=root_dir, train=False, transform=transform, download=True)

        super().__init__(
            train_dataset=train_dataset,        # size = 60000
            test_dataset=test_dataset,          # size = 10000
            label_names=train_dataset.classes   # = ['????']
        )

    # @functools.lru_cache
    # def restricted_data(self, up_to_label: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    #     res_train_dataset = Subset(
    #         self.train_dataset, [i for i, label in enumerate(self.train_dataset.targets) if label<up_to_label])
    #     res_test_dataset = Subset(
    #         self.test_dataset, [i for i, label in enumerate(self.test_dataset.targets) if label<up_to_label])
    #     return res_train_dataset, res_test_dataset
    #
    # @functools.lru_cache
    # def train_loader(self, up_to_label:int, batch_size: int) -> DataLoader:
    #     train_dataset, _ = self.restricted_data(up_to_label)
    #     return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #
    # @functools.lru_cache
    # def test_loader(self, up_to_label:int, batch_size: int = -1) -> DataLoader:
    #     _, test_dataset = self.restricted_data(up_to_label)
    #     batch_size = batch_size if batch_size > 0 else len(test_dataset)
    #     return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
        if isinstance(self.hidden, int):
            self.hidden = (self.hidden,)
        assert all([d>0 for d in self.hidden])
        assert self.batch_size > 0
        assert self.epochs > 0

    def update_model(self, model_params: ModelParameters):
        self.seed = model_params.seed
        self.hidden = model_params.hidden[1:-1]

    def update_image_data(self, data_parameters: ImageDataParameters):
        self.batch_size = data_parameters.batch_size
        self.up_to_label = data_parameters.up_to_label

    def update_training(self, training_parameters: TrainingParameters):
        self.learning_rate = training_parameters.learning_rate
        self.epochs = training_parameters.epochs

params = MnistParameters(
    seed=1,                     # model
    up_to_label=2,             # data
    learning_rate=0.001,        # training
    hidden=(),                  # model
    batch_size=128,             # data
    epochs=2)                   # training

# <editor-fold desc=" ------------------------ Prepare data ------------------------">

database = MnistData()

train_data, test_data = database.restricted_data(up_to_label=params.up_to_label)
train_size = len(train_data)
train_loader = database.train_loader(up_to_label=params.up_to_label, batch_size=params.batch_size)
test_loader = database.test_loader(up_to_label=params.up_to_label)

# Prepare model
torch.manual_seed(params.seed)
model = SimpleClassifier(dimensions=[28 * 28] + list(params.hidden) + [params.up_to_label])

# </editor-fold>

def plot_layer0_filters(model: SimpleClassifier):
    arr = model.get_linear_layer(0).weight.detach().cpu().numpy()
    arr = arr.reshape(arr.shape[0], 28, 28)
    visualization.plot_images(arr)


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

run_scatter_animation = False
if run_scatter_animation:
    """
    Create scatter animation of the outputs of all the linear layers going into dimension 2.
    The frames corresponds to each learning step
    """

    with torch.no_grad():
        for X_test, Y_test in test_loader:
            break  # because I hate python!!!

    #                                         n x 1 x 28 x 28               n
    collector = OutputProgressCollector(model=model, test_input=X_test, test_output=Y_test)

    train_model(model=model, data_loader=train_loader, train_size=train_size,
                epochs=params.epochs, learning_rate=params.learning_rate,
                train_step_callback=collector.progress_callback)

    hidden_str = '_'.join(str(d) for d in params.hidden)
    collector.generate_animation()  # save_animation_file=f'media/labels_{params.up_to_label}_hidden_{hidden_str}')


# </editor-fold>

# <editor-fold desc=" ------------------------ Plot filters from first layer ------------------------">
#
def plot_first_layer_images(model: SimpleClassifier, title:str, save_path: str = ''):
    """
    Plots the filters of the first layer after each epoch and show them
    Change 'save_path' to a file path in order to save the output (instead of showing it in plt).
    """
    special_case = False
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
            if special_case:
                ax.set_title(f'{weights[0][i]:.3f} ; {weights[1][i]:.3f}\nBias={b:.3f}', fontsize=12)
            else:
                ax.set_title(f'Bias={b}', fontsize=12)
    if save_path != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def callback_to_filters(epoch: int, num_data_points: int):
    if num_data_points>=0:     # only run when each epoch is finished
        return
    plot_first_layer_images(f'Epoch {epoch}') #, save_path = f'media/filters/seed{params.seed}/filters_{epoch}')

run_filter_plots = False
if run_filter_plots:
    train_model(model=model, data_loader=train_loader, train_size=train_size,
                epochs=params.epochs, learning_rate=params.learning_rate,
                train_step_callback=callback_to_filters)


# </editor-fold>

# <editor-fold desc=" ------------------------ Everything together ------------------------">


run_analyze_outputs = False
if run_analyze_outputs:

    # train_loader = DataLoader(Subset(train_data, range(10))  , batch_size=1, shuffle=True)

    train_model(model=model, data_loader=train_loader, train_size=train_size,
                epochs=params.epochs, learning_rate=params.learning_rate)

    weights0 = model.get_linear_layer(0).weight.detach().cpu().numpy()
    bias0 = model.get_linear_layer(0).bias.detach().cpu().numpy()
    weights0 = weights0.reshape(weights0.shape[0], 28, 28)

    linear_outputs = model.collect_output() # add hooks to collect output of linear layers
    model.eval()
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            model_output = model(X_test)
            break  # because I hate python!!!

    print('Train data:')
    train_error_rate = test_model(model, train_loader)
    print('Test data:')
    test_error_rate = test_model(model, test_loader)

    hidden_length = len(params.hidden)

    if hidden_length == 1:
        weights1 = model.get_linear_layer(1).weight.detach().cpu().numpy()
        visualization.plot_more_filters(
            weights0=weights0, bias0=bias0, weights1=weights1,
            inputs=X_test.numpy().squeeze(1), labels=Y_test,
            outputs0=linear_outputs[model.get_linear_layer(0)],
            outputs1=linear_outputs[model.get_linear_layer(1)],
            )
    else:
        outputs_last = linear_outputs[model.get_linear_layer(hidden_length)]
        visualization.plot_more_filters2(
            weights0=weights0, bias0=bias0,
            inputs=X_test.numpy().squeeze(1), labels=Y_test,
            outputs_first=linear_outputs[model.get_linear_layer(0)],
            outputs_last=outputs_last,
            )


# </editor-fold>


T = TypeVar('T')
IteratorFactory = Callable[[],Iterator[T]]

run_save_errors = True

up_to_label = 2
running_data = XpData(
    file_path=f'mnist_data{up_to_label}.csv',
    data_cls=analyse_data.Parameters,
    values = dict(train_error_rate=float, test_error_rate=float))

# database.plot_examples()

# analyse_data.save_errors(
#     running_data=running_data, test_loader=test_loader,
#     model_generator = lambda: SimpleClassifier.generate_from_parameters(
#         seeds                   =[1, 10, 100, 1000],
#         linear_layer_dimensions =[(28*28,)+hidden+(up_to_label,) for hidden in [()]]),
#     data_generator = lambda: database.train_loaders_from_parameters(
#         up_to_label             =[up_to_label],
#         batch_sizes             =[64]
#     ),
#     training_generator = lambda: TrainingParameters.from_parameters(
#         learning_rates          =[0.001],
#         epoch_list              =[5])
# )

# analyse_data.errors_per_epoch(
#     running_data,
#     # seed = 100,
#     learning_rate=0.001,
#     hidden=(),
#     batch_size=64
# )


# for X_test, Y_test in test_loader:
#     break  # because I hate python!!!
# Y_test = Y_test.detach().numpy()
# X_test = X_test.detach().numpy()
# X_test = X_test.reshape(X_test.shape[0], -1)

# arr = np.squeeze(X_test[Y_test == 0])

# visualization.plot_images(
#     arr = arr[:20],
# )
# visualization.pca_visualize(arr, 10)

