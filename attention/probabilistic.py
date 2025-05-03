import numpy as np
import torch
from typing import List

class ProbModel:
    """
    The full probabilistic model of a text, namely
    	model(c1,...,cn) = (p1,...,pk)
    where pi is the probability of seeing the i-th letter after seeing the sequence c1...cn .
    To avoid absolute zero, we add an epsilon weight when counting the frequencies. In particular,
    if c1...cn doesn't appear at all, then the resulting probability vector will be the uniform probability.
    """

    def __init__(self, abc_size: int, window_size: int):
        self.window_size = window_size
        self.abc_size = abc_size
        self.probabilities = np.zeros(shape=(self.abc_size,)*(self.window_size+1))

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.window_size
        # indices = tuple(np.moveaxis(input, -1, 0))   # for numpy array
        indices = tuple(torch.movedim(input, -1, 0))
        return torch.from_numpy(self.probabilities[indices])

    def prob(self, input: torch.Tensor) -> torch.Tensor:
        return self(input)

    def train(self, X, Y, epsilon: float = 0.001):
        assert len(X) == len(Y)
        assert X.shape[1] == self.window_size

        # self.probabilities[c1, ..., c_win_size, c] =
        #   probability of seeing c after (c1, ..., c_win_size)
        self.probabilities = np.zeros(shape=(self.abc_size,)*(self.window_size+1))
        np.add.at(self.probabilities, tuple(np.concatenate([X, Y[:, np.newaxis]], axis = 1).T), 1)
        self.probabilities += epsilon
        self.probabilities /= self.probabilities.sum(axis = -1, keepdims=True)


    def generate(self, initial_message: List[int], num_letters: int):
        assert len(initial_message) == self.window_size

        indices = [i for i in initial_message]
        if self.window_size == 0:
            for _ in range(num_letters):
                next_letter = np.random.choice(self.abc_size, p=self.probabilities)
                indices.append(next_letter)

            return indices

        for _ in range(num_letters):
            window_weights = self.probabilities[tuple(indices[-self.window_size:])]
            next_letter = np.random.choice(self.abc_size, p=window_weights)
            indices.append(next_letter)

        return indices