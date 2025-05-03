import math

import matplotlib.pyplot as plt

from general import model_tester
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Dict, List, Iterator, Tuple, Optional
import torch
import torch.nn as nn
from vision.misc import cached, cached2
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from attention.text_data import TextData
from attention.probabilistic import ProbModel

class SimpleModel(nn.Module):

    def __init__(self, abc_size: int, letter_dim: int, window_size: int, hidden: int):
        super(SimpleModel, self).__init__()

        self.abc_size = abc_size
        self.letter_dim = letter_dim
        self.window_size = window_size
        self.hidden = hidden

        self.embedding = nn.Linear(abc_size, letter_dim, bias = False)
        self.linear1 = nn.Linear(window_size * letter_dim, hidden)
        self.linear2 = nn.Linear(hidden, abc_size)
        # self.softmax = nn.Softmax(dim=1)

        self.initial_state = self.state_dict()

    def initialize_state(self):
        self.load_state_dict(self.initial_state)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(x), dim=1)

    def forward(self, x):
        # x is an (n, windows_size) array, with values in [0,..., abc_size)
        x = F.one_hot(x, self.abc_size).float()         # (n, window_size, abc_size)
        x = self.embedding(x)                           # (n, window_size, letter_dim)
        x = x.view(-1, self.window_size * self.letter_dim)  #  (n, window_size * letter_dim)
        x = self.linear1(x)                             # (n, hidden)
        x = torch.tanh(x)                               # (n, hidden)
        x = self.linear2(x)                             # (n, abc_size)
        # x = self.softmax(x)                             # (n, abc_size)

        return x

    def generate(self, initial_message: List[int], num_letters: int):
        assert len(initial_message) == self.window_size

        indices = [i for i in initial_message]
        self.eval()

        for _ in range(num_letters):
            input = torch.tensor(indices[-self.window_size:])
            window_weights = self.prob(input).view(-1).detach().numpy()
            next_letter = np.random.choice(self.abc_size, p=window_weights)
            indices.append(next_letter)

        return indices

file_name = 'alice'
total_letter_count = 10000
initial_text = 'alice'
window_size = 5
initial_text = initial_text[:window_size]
seed = 0

processed_text = TextData(file_name=file_name, total_letter_count=total_letter_count)

# <editor-fold desc=" ------------------------ Pair probabilities ------------------------">


#
# model = SimpleModel(text_data.abc_size, letter_dim=3, window_size=1, hidden=100)
# X, Y = text_data.get_data(1)
# dataset = TensorDataset(torch.from_numpy(X.copy()).long(), torch.from_numpy(Y).long())
# train_sample_size = len(dataset)
# train_loader=DataLoader(dataset, batch_size=32, shuffle=False)
# model_tester.train_model(
#     model=model,
#     data_loader=train_loader,
#     train_size=train_sample_size,
#     parameters = model_tester.TrainingParameters(learning_rate=math.exp(-5), epochs=5)
# )
#
# view_pair_probabilities(text_data, model = model)
# exit(0)

# </editor-fold>

X, Y = processed_text.get_data(window_size)

dataset = TensorDataset(torch.from_numpy(X.copy()).long(), torch.from_numpy(Y).long())

train_dataset, dev_dataset, text_dataset = torch.utils.data.random_split(dataset, lengths=[0.8, 0.1, 0.1])
train_sample_size = len(train_dataset)

# model = ProbModel(text_data.abc_size, window_size)
# model.train(X, Y)

model = SimpleModel(processed_text.abc_size, 2, window_size, 100)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=False)
dev_loader=DataLoader(dev_dataset, batch_size=1, shuffle=False)

def optimize_learning_rate(model: nn.Module,train_loader, train_sample_size, dev_loader):
    learning_rates = np.arange(-10,2, 0.5)
    train_error_rates = []
    dev_error_rates = []
    for learning_rate in learning_rates:
        print(f'\n-------------------------')
        print(f'Training on learning rate {math.exp(learning_rate)}')
        model.initialize_state()

        model_tester.train_model(
            model=model,
            data_loader=train_loader,
            train_size=train_sample_size,
            parameters = model_tester.TrainingParameters(learning_rate=math.exp(learning_rate), epochs=1)
        )

        error_rate = model_tester.test_model(
            model=model,
            test_loader=train_loader
        )
        train_error_rates.append(error_rate)

        error_rate = model_tester.test_model(
            model=model,
            test_loader=dev_loader
        )
        dev_error_rates.append(error_rate)

    plt.plot(learning_rates, train_error_rates, label='Train')
    plt.plot(learning_rates, dev_error_rates, label='Dev')
    plt.legend()
    plt.xlabel(r'$\log(\text{Learning Rate})$')
    plt.ylabel(r'Error rate')
    plt.show()

# optimize_learning_rate(model, train_loader, train_sample_size, dev_loader)

# exit()


