import math

import matplotlib.pyplot as plt

from general import model_tester
import numpy as np
from typing import Dict, List, Iterator, Tuple, Optional
import torch
import torch.nn as nn
from vision.misc import cached, cached2
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from attention.text_data import TextData

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

class Transformer(nn.Module):

    def __init__(self, abc_size: int, letter_dim: int, window_size: int):
        super(Transformer, self).__init__()

        self.abc_size = abc_size
        self.letter_dim = letter_dim
        self.window_size = window_size

        self.embedding = nn.Linear(abc_size, letter_dim, bias = False)
        self.positional_embedding = nn.Linear(window_size, letter_dim, bias = False)

        self.head_size = 16
        self.key = nn.Linear(letter_dim, self.head_size)
        self.query = nn.Linear(letter_dim, self.head_size)
        self.value = nn.Linear(letter_dim, self.abc_size)

    def forward(self, x: torch.Tensor):
        # 1. Embedding: discrete [0,...,num_tokens-1] to linear R^num_tokens
        # 2. Token dimension: linear to R^d
        # 3. Position data: add data according to position in text (still in R^d)
        # 4. Attention: Combine data with previous tokens

        pos = torch.eye(window_size)                        # (window_size, window_size)
        pos = self.positional_embedding(pos)                # (window_size, letter_dim)
        position_vectors = self.positional_embedding

        # x is an (n, windows_size) array, with values in [0,..., abc_size)
        x = F.one_hot(x, self.abc_size).float()             # (n, window_size, abc_size)
        x = self.embedding(x)                               # (n, window_size, letter_dim)
        x += pos                                            # (n, window_size, letter_dim)

        keys: torch.Tensor = self.key(x)                    # (n, window_size, self.head_size)
        queries = self.query(x)                             # (n, window_size, self.head_size)
        values = self.value(x)                              # (n, window_size, self.abc_size??)
        weights = queries @ keys.transpose(-2, -1)          # (n, window_size, window_size)

        connections = torch.tril(torch.ones(window_size, window_size))
        weights = weights.masked_fill(connections == 0, float('-inf'))
        weights /= self.head_size**-0.5          # normalization
        weights = F.softmax(weights, dim = -1)              # (n, window_size, window_size)

        x = weights @ values                                # (n, window_size, self.abc_size??)

        return x.transpose(-2, -1)                  # (n, self.abc_size, window_size)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        output = self(x)[..., :, -1]                        # (n, self.abc_size)
        return F.softmax(output, dim=-1)

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

window_size = 4
processed_text = TextData(file_name=file_name)#, total_letter_count=total_letter_count)
# Preparing the data
X, Y = processed_text.get_data(window_size=window_size)

# to torch
X = torch.from_numpy(X.copy()).long() # The numpy array X is not writable, so we need to copy it before converting it to a pytorch array
Y = torch.from_numpy(Y).long()

dataset = TensorDataset(X, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.85, 0.15])

train_sample_size = len(train_dataset)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)

window_size = 4
from attention.probabilistic import ProbModel
prob_model4 = ProbModel(abc_size=processed_text.abc_size, window_size=window_size)
prob_model4.train(X, Y, epsilon=0.0001)
test_error_rate = model_tester.test_model(
    model=prob_model4,
    test_loader=test_loader
)

# <editor-fold desc=" ------------------------ Pair probabilities ------------------------">


#
# model = Transformer(processed_text.abc_size, letter_dim=3, window_size=window_size)
# X, _ = processed_text.get_data(window_size=window_size)
# X = torch.from_numpy(X.copy()).long()
# Y = X[1:]
# X = X[:-1]
# dataset = TensorDataset(X, Y)
# # train_dataset, dev_dataset, text_dataset = torch.utils.data.random_split(dataset, lengths=[0.8, 0.1, 0.1])
# train_dataset = dataset
# train_sample_size = len(train_dataset)
# train_loader=DataLoader(train_dataset, batch_size=32, shuffle=False)
# # dev_loader=DataLoader(dev_dataset, batch_size=1, shuffle=False)
# model_tester.train_model(
#     model=model,
#     data_loader=train_loader,
#     train_size=train_sample_size,
#     parameters = model_tester.TrainingParameters(learning_rate=math.exp(-6), epochs=500)
# )
#
# generated_int_message = model.generate(initial_message=processed_text.to_int(initial_text), num_letters=500)
# generated_text_message = processed_text.to_text(generated_int_message)
# print(generated_text_message)





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


