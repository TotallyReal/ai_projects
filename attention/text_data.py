import numpy as np
from typing import List, Tuple
from numpy.lib.stride_tricks import sliding_window_view

class TextData:

    def __init__(self, file_name: str, total_letter_count: int = -1):
        with open(f'data/{file_name}.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
            if total_letter_count == -1:
                total_letter_count = len(self.text)
            self.text = self.text[:min(total_letter_count, len(self.text))].lower()

        self.alphabet = sorted(list(set(self.text)))
        self.abc_size = len(self.alphabet)
        self.letter_to_index = {c: i for i, c in enumerate(self.alphabet)}
        self.index_to_letter = {i: c for c, i in self.letter_to_index.items()}

        self.int_text = np.array([self.letter_to_index[c] for c in self.text])

    def get_data(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        X = sliding_window_view(self.int_text[:-1], window_size)
        Y = self.int_text[window_size:]

        return X, Y

    def to_text(self, int_message: List[int]) -> str:
        return ''.join(self.index_to_letter[i] for i in int_message)

    def to_int(self, initial_text: str) -> np.ndarray:
        return np.array([self.letter_to_index[c] for c in initial_text])