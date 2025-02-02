from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Callable

from misc import print_progress_bar
from misc import time_me

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def empty_callback(epoch: int, num_data_points: int):
    pass


@time_me
def train_model(
        model: nn.Module, data_loader: DataLoader, train_size: int = 0,
        epochs:int = 10, learning_rate: float = 0.001,
        train_step_callback: Callable[[int, int],None] = empty_callback):
    """
    Runs the model on the given data, with the given epochs and learning rate.
    If train_size > 0, will print a progress bar during each epoch.

    The train_step_callback(current_epoch, num_data_points) will be called after each step, where
    the number of data points is counted for the current epoch only.
    At the finish of each epoch it will be called with num_data_points = -1
    """
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        data_so_far = 0
        if train_size>0:
            print_progress_bar(iteration=data_so_far, total=train_size)
        for X_train, Y_train in data_loader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            # Forward pass
            Y_pred = model(X_train)
            loss = loss_function(Y_pred, Y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_so_far += X_train.shape[0]

            train_step_callback(epoch, data_so_far)

            if train_size>0:
                # TODO: consider moving this into a callback
                print_progress_bar(iteration=data_so_far, total=train_size)
        train_step_callback(epoch, -1)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")



# Testing the model
def test_model(model: nn.Module, test_loader: DataLoader):
    model.eval()
    total_errors = 0
    total_samples = 0

    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)

            outputs = model(X_test)
            predicted = torch.argmax(outputs, dim=1)

            total_errors += (predicted != Y_test).sum().item()
            total_samples += len(Y_test)

        error_rate = total_errors/total_samples
        print(f"Number of errors: {total_errors}/{total_samples} ~ {error_rate}")

    return error_rate

