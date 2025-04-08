from dataclasses import dataclass, field
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Dict, List, Iterator, Tuple, Optional
import torch
from vision.misc import cached, cached2


import torch.nn as nn
LETTER_COUNT = 20000

class TextData:

    def __init__(self, file_name: str, total_letter_count: int):
        with open(f'data/{file_name}.txt', 'r', encoding='utf-8') as f:
            text = f.read()[:total_letter_count].lower()

        self.alphabet = sorted(list(set(text)))
        self.abc_size = len(self.alphabet)
        self.letter_to_index = {c: i for i, c in enumerate(self.alphabet)}
        self.index_to_letter = {i: c for c, i in self.letter_to_index.items()}

        self.int_text = np.array([self.letter_to_index[c] for c in text])

    def get_data(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        X = sliding_window_view(self.int_text[:-1], window_size)
        Y = self.int_text[window_size:]

        return X, Y

    def to_text(self, int_message: List[int]) -> str:
        return ''.join(self.index_to_letter[i] for i in int_message)

    def to_int(self, initial_text: str) -> np.ndarray:
        return np.array([self.letter_to_index[c] for c in initial_text])


class ProbModel:

    def __init__(self, window_size: int, abc_size: int):
        self.window_size = window_size
        self.abc_size = abc_size
        self.probabilities = np.zeros(shape=(self.abc_size,)*(self.window_size+1))

    def train(self, X, Y):
        assert len(X) == len(Y)
        assert X.shape[1] == self.window_size
        # counters = np.zeros(shape=(self.abc_size,)*(self.window_size+1))
        self.probabilities = np.zeros(shape=(self.abc_size,)*(self.window_size+1))
        np.add.at(self.probabilities, tuple(np.concatenate([X, Y[:, np.newaxis]], axis = 1).T), 1)
        self.probabilities += 0.001
        self.probabilities /= self.probabilities.sum(axis = -1, keepdims=True)

    def generate(self, initial_message: List[int], num_letters: int):
        assert len(initial_message) == self.window_size

        indices = [i for i in initial_message]

        for _ in range(num_letters):
            window_weights = self.probabilities[tuple(indices[-self.window_size:])]
            next_letter = np.random.choice(self.abc_size, p=window_weights)
            indices.append(next_letter)

        return indices

file_name = 'alice'
total_letter_count=10000
window_size=4
initial_text = 'alic'
seed = 0

text_data = TextData(file_name=file_name, total_letter_count=total_letter_count)
X, Y = text_data.get_data(window_size)

model = ProbModel(window_size, text_data.abc_size)
model.train(X, Y)

initial_message = text_data.to_int(initial_text)
np.random.seed(seed)
generated_message = model.generate(initial_message, num_letters=1000)
print(text_data.to_text(generated_message))
