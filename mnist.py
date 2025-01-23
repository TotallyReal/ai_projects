from misc import Timer

timer = Timer()
timer.time(print_time=False)

from typing import List
import globals
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xp_data import XpData
from model_tester import test_model, train_model


timer.time(msg='finish imports')


class SimpleTransformModel(nn.Module):
    def __init__(self, m: int, hidden_layers: List[int], k: int):
        super(SimpleTransformModel, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        all_layers_sizes = [m*m] + hidden_layers + [k]
        for input_size, output_size in zip(all_layers_sizes[:-1], all_layers_sizes[1:]):
            layers += [
                nn.Linear(input_size, output_size),
                nn.Sigmoid()
            ]

        layers[1] = nn.ReLU()

        self.linear_relu = nn.Sequential(
            *layers[:-1]    # don't add ReLu before the last softmax
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu(x)
        x = self.softmax(x)
        return x


# Example usage
if __name__ == "__main__":
    seed = 1

    # <editor-fold desc=" ------------------------ Prepare Data ------------------------">

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.ToTensor()  # Convert to tensor and normalize to [0, 1]
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
    test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=64, shuffle=False)

    resolution = 28

    # </editor-fold>


    # Train the model
    learning_rate = 0.001
    data_index = dict(seed=seed, lr=learning_rate)

    running_data = XpData(file_path='running_data.csv', index_cols=globals.data_parameters, values_cols=globals.data_output)


    timer.time(msg='start loop')
    for hidden_layers in [[9]]:
        torch.manual_seed(seed)
        model = SimpleTransformModel(m=resolution, hidden_layers=hidden_layers, k=10)
        initial_state = model.state_dict()
        data_index['hidden'] = tuple(hidden_layers)
        for epochs in range(1, 4):
            data_index['epochs'] = epochs
            for batch_size in [32, 64]:
                print('----------------------------------------------------')
                data_index['batch_size'] = batch_size
                # if running_data.contains(data_index):
                #     print(f'Already contains {data_index}')
                #     continue

                print(f'Running for parameters {data_index=}')

                model.load_state_dict(initial_state)
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

                train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate)
                print('Train data:')
                train_error_rate = test_model(model, train_loader)
                print('Test data:')
                test_error_rate = test_model(model, test_loader)

                # running_data.add_entry(
                #     **data_index,
                #     train_error_rate=train_error_rate, test_error_rate=test_error_rate)

    timer.time(msg='end loop')
    # running_data.save()
