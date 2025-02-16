from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import Callable, Iterator, Tuple, TypeVar
import torch
from torch.utils.data import DataLoader
from general.model_tester import TrainingParameters
from general.image_data import ImageDataParameters, ImageData
from general.xp_data import XpData
from general.models import SimpleClassifier, ModelParameters
from general.model_tester import test_model, train_model
# from mnist.mnist_exp import MnistParameters

T = TypeVar('T')
IteratorFactory = Callable[[],Iterator[T]]


@dataclass
class Parameters:
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

def run_hyper_parameters(
        model_generator:    IteratorFactory[ Tuple[torch.nn.Module, ModelParameters] ],
        data_generator:     IteratorFactory[ Tuple[DataLoader, ImageDataParameters] ],
        training_generator: IteratorFactory[ TrainingParameters ])\
        -> Tuple[torch.nn.Module, Parameters, DataLoader]:
    params = Parameters()
    for model, model_params in model_generator():
        params.update_model(model_params)
        for train_loader, image_data_params in data_generator():
            params.update_image_data(image_data_params)
            for training_params in training_generator():
                params.update_training(training_params)

                model.initialize_state()
                yield model, params, train_loader


def save_errors(
        running_data: XpData, test_loader: DataLoader,
        model_generator:    IteratorFactory[ Tuple[torch.nn.Module, ModelParameters] ],
        data_generator:     IteratorFactory[ Tuple[DataLoader, ImageDataParameters] ],
        training_generator: IteratorFactory[ TrainingParameters ],
        save_each_epoch: bool = True):

    params = Parameters()
    for model, model_params in model_generator():
        params.update_model(model_params)
        for train_loader, image_data_params in data_generator():
            params.update_image_data(image_data_params)
            for training_params in training_generator():
                params.update_training(training_params)

                # Ignore completely if this case is already in the database
                print('----------------------------------------------------')
                if running_data.contains_index(params):
                    print(f'Skipping {params}')
                    continue
                print(f'Running for parameters {params=}')

                model.initialize_state()

                def epoch_save_callback(current_epoch: int, num_data_points: int):
                    if num_data_points >= 0:
                        return # only run on epochs

                    # Ignore this epoch if already in the database
                    params.epochs = current_epoch + 1
                    if running_data.contains_index(params):
                        print(f'Skipping {params}')
                        return

                    model.eval()

                    print('Train data:')
                    train_error_rate = test_model(model, train_loader)
                    print('Test data:')
                    test_error_rate = test_model(model, test_loader)

                    model.train()

                    # Add entry, and save to file if already created a lot of entries
                    running_data.add_entry(
                        params,
                        train_error_rate=train_error_rate, test_error_rate=test_error_rate)


                train_model(
                    model=model,
                    data_loader=train_loader, train_size=image_data_params.train_size,
                    parameters=training_params,
                    train_step_callback=epoch_save_callback if save_each_epoch else None)

    running_data.save()



def errors_per_epoch(
        data: XpData,
        seed: int = 1,
        learning_rate: float = 0.001,
        hidden: Tuple[int, ...] = (),
        batch_size: int = 64):
    # 1. Extract rows where 'hidden' is an empty tuple
    filtered_df = data.df[data.df['hidden'].apply(lambda x: x == hidden)]
    filtered_df = filtered_df[filtered_df['seed'] == seed]
    filtered_df = filtered_df[filtered_df['batch_size'] == batch_size]
    filtered_df = filtered_df[filtered_df['learning_rate'] == learning_rate]

    # 2. Extract relevant columns
    epochs = filtered_df['epochs']
    train_error = filtered_df['train_error_rate']
    test_error = filtered_df['test_error_rate']

    # 3. Plot the error rates
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_error, label="Train Error Rate", color='blue')
    plt.plot(epochs, test_error, label="Test Error Rate", color='red')

    x_max = max(epochs)
    plt.xticks(range(1, x_max, 1))
    plt.xlim(1,x_max)

    # Labels and legend
    plt.xlabel("Epochs")
    plt.ylabel("Error Rate")
    plt.title("Train vs Test Error Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
#
#
# running_data = XpData(
#     file_path=f'mnist/running_data{10}.csv',
#     data_cls=MnistParameters,
#     values = dict(train_error_rate=float, test_error_rate=float))
# errors_per_epoch(running_data)



# from xp_data import XpData
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# running_data = XpData(file_path='running_data2.csv', index_cols=data_parameters, values_cols=data_output)
# df = running_data.df
#
#
# def filter(df, partial_data) -> pd.DataFrame:
#     mask = pd.Series(True, index=df.index)
#     for col, value in partial_data.items():
#         mask &= (df[col] == value)
#     return df[mask]
#
#
#
# def show_errors(df, hidden_layers):
#     # df = df[df.index.get_level_values('hidden') == hidden_layers]
#     # df = df[df['hidden'] == hidden_layers]
#     df = df.xs(key=(hidden_layers), level=('hidden'))
#
#     combined_colors = np.concatenate([df['train_error_rate'], df['test_error_rate']])
#     norm = plt.Normalize(vmin=combined_colors.min(), vmax=combined_colors.max())
#
#     plt.scatter(
#         df.index.get_level_values('epochs')-0.1, df.index.get_level_values('batch_size'),
#         c=df['train_error_rate'],
#         cmap='viridis', s=100, norm=norm)
#     scatter = plt.scatter(
#         df.index.get_level_values('epochs')+0.1, df.index.get_level_values('batch_size'),
#         c=df['test_error_rate'],
#         cmap='viridis', s=100, norm=norm)
#
#     # Add a colorbar to indicate the y value
#     colorbar = plt.colorbar(scatter)
#     colorbar.set_label('error rate')
#
#     # Labels and title
#     plt.xlim(0, df.index.get_level_values('epochs').max()+1)
#     plt.xlabel('epochs')
#     plt.ylabel('batch_size')
#     plt.title(f'train \\ test error rate {hidden_layers}')
#
#     # Show the plot
#
# plt.figure(figsize=(12, 9))
# # for k in range(6):
# #     plt.subplot(2, 3, k+1)
# #     show_errors(df,str((k+1,)))
# # plt.show()
#
#
# df = filter(df,{'batch_size':64, 'hidden':str((2,2,2))})
# print(df)