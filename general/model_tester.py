from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Callable, List, Iterator, Optional
import torch.nn.functional as F

from misc import print_progress_bar
from misc import time_me

def get_device(verbose=False):
    """
    Returns the best available PyTorch device.
    Priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple MPS device (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device

def empty_callback(epoch: int, num_data_points: int, current_loss):
    pass

@dataclass
class TrainingParameters:
    learning_rate: float = 0.001
    epochs: int = 1

    def __post_init__(self):
        assert self.learning_rate > 0
        assert self.epochs > 0

    @staticmethod
    def from_parameters(learning_rates: List[float], epoch_list: List[int]) -> Iterator['TrainingParameters']:
        for learning_rate in learning_rates:
            for epoch in epoch_list:
                yield TrainingParameters(learning_rate=learning_rate, epochs=epoch)

default_training_parameters = TrainingParameters(
    epochs = 10,
    learning_rate = 0.001
)

TrainStepCallback = Callable[[int, int, float],None]

@time_me
def train_model(
        model: nn.Module, data_loader: DataLoader, train_size: int = 0,
        parameters: TrainingParameters = default_training_parameters,
        loss_function = nn.CrossEntropyLoss(),
        train_step_callback: TrainStepCallback = empty_callback):
    """
    Runs the model on the given data, with the given epochs and learning rate.
    If train_size > 0, will print a progress bar during each epoch.

    The train_step_callback(current_epoch, num_data_points, loss) will be called after each step, where
    the number of data points is counted for the current epoch only.
    At the finish of each epoch it will be called with num_data_points = -1, and the total loss of that
    epoch.
    """
    device = get_device()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate)

    model.train()

    for epoch in range(parameters.epochs):
        num_samples = 0
        total_loss = 0.0
        if train_size>0:
            print_progress_bar(iteration=num_samples, total=train_size)
        for X_train, Y_train in data_loader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            batch_size = len(Y_train)

            # Forward pass
            Y_pred = model(X_train)
            loss = loss_function(Y_pred, Y_train)
            total_loss += loss.item() * batch_size

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_samples += batch_size

            train_step_callback(epoch, num_samples, loss.item())

            if train_size>0:
                # TODO: consider moving this into a callback
                print_progress_bar(iteration=num_samples, total=train_size)
        train_step_callback(epoch, -1, total_loss / num_samples)

        print(f"Epoch [{epoch+1}/{parameters.epochs}], Average Loss: {total_loss / num_samples:.4f}")

# <editor-fold desc=" ------------------------ Loss functions ------------------------">

# output of size (n,m)
# predictions of size (n) with values in {0,...,m-1}
LossFunction = Callable[[torch.Tensor, torch.Tensor], float]

def error_rate_argmax(output: torch.Tensor, predictions: torch.Tensor) -> float:
    return (torch.argmax(output, dim=1) != predictions).sum().item()/len(predictions)

def error_rate_prob(output: torch.Tensor, predictions: torch.Tensor) -> float:
    prob = F.softmax(output, dim=1)
    n = len(prob)
    return 1 - prob[torch.arange(n), predictions].sum().item()/n

# </editor-fold>



# Testing the model
def test_model(
        model: nn.Module, test_loader: DataLoader,
        loss_function: Optional[LossFunction] = None):
    if loss_function is None:
        loss_function = error_rate_argmax

    device = get_device()
    model = model.to(device)

    model.eval()
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)

            outputs = model(X_test)
            batch_size = Y_test.size(0)

            total_loss += batch_size * loss_function(outputs, Y_test)
            total_samples += batch_size

    return total_loss/total_samples

