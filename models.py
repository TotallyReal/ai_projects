import numpy as np
from typing import Dict, List, Iterator
import torch
import torch.nn as nn

class RotationLayer(nn.Module):
    def __init__(self, theta: float):
        super().__init__()
        self.theta = torch.tensor(theta, dtype=torch.float32)  # Store as tensor
        self.register_buffer("rotation_matrix", self._get_rotation_matrix(self.theta))

    def _get_rotation_matrix(self, theta: torch.Tensor):
        """Create a 2x2 rotation matrix for the given angle (in degrees)."""
        return torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]], dtype=torch.float32)

    def theta_str(self):
        return f'2π*{(self.theta/(2*torch.pi)):.3f}'

    def forward(self, x):
        return torch.matmul(x, self.rotation_matrix.T)  # Apply rotation

class LearnableRotationLayer(nn.Module):
    def __init__(self, theta: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))  # Learnable angle in radians

    def theta_str(self):
        return f'2π*{(self.theta/(2*torch.pi)):.3f}'

    def forward(self, x):
        theta = self.theta  # Keep theta in radians
        rotation_matrix = torch.stack([
            torch.cos(theta), -torch.sin(theta),
            torch.sin(theta), torch.cos(theta)
        ]).reshape(2, 2)
        return torch.matmul(x, rotation_matrix.T)

class SimpleClassifier(nn.Module):
    """
    A simple module containing linear layers seperated by ReLu nonlinear layers, with final Softmax:
    Linear -> ReLu -> Linear -> ReLu -> ... -> Linear -> SoftMax
    """

    def __init__(self, dimensions: List[int]):
        """
        The k-th linear relu_layers (starting from 0) goes from dimensions[k] to dimensions[k+1].
        In particular, the input of the classifier is dimensions[0] (which is automatically flatten),
        and output is dimensions[-1].
        """
        assert all(d > 0 for d in dimensions)
        super(SimpleClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self._linear_layers = []

        relu_layers = []
        linear_index = 0
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            linear_layer = nn.Linear(input_size, output_size)
            linear_layer.name = f'Linear_{linear_index}'
            linear_index += 1
            self._linear_layers.append(linear_layer)
            relu_layers += [
                linear_layer,
                nn.ReLU()
            ]

        self.linear_relu = nn.Sequential(
            *relu_layers[:-1]    # don't add ReLu before the last softmax
        )

        self.softmax = nn.Softmax(dim=1)

    def get_linear_layer(self, k: int = 0) -> nn.Linear:
        return self._linear_layers[k]

    def number_of_linear_layers(self) -> int:
        return len(self._linear_layers)

    def linear_layers(self) -> Iterator[nn.Linear]:
        return iter(self._linear_layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu(x)
        x = self.softmax(x)
        return x

    def collect_output(self) -> Dict[torch.nn.Linear, np.ndarray]:
        """
        Creates an output collector for the linear layers
        """
        outputs = dict()

        def _save_output(layer, input, output):
            outputs[layer] = output.detach().numpy()  # batch_size x dim_output

        for linear_layer in self.linear_layers():
            linear_layer.register_forward_hook(_save_output)

        return outputs