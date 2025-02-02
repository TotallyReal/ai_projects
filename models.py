from typing import List, Iterator
import torch.nn as nn

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
        self.progress_output = None
        self.flatten = nn.Flatten()
        self.outputs = dict()
        self._linear_layers = []
        relu_layers = []
        linear_index = 0
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            linear_layer = nn.Linear(input_size, output_size)
            linear_layer.name = f'Linear_{linear_index}'
            linear_index += 1
            self._linear_layers.append(linear_layer)
            # linear_layer.register_forward_hook(self.output_hook)
            relu_layers += [
                linear_layer,
                nn.ReLU()
            ]

        # relu_layers[1] = nn.ReLU()

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

    def output_hook(self, module, input, output):
        self.outputs[module.name] = output

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu(x)
        x = self.softmax(x)
        return x

    def get_tracked_info(self):
        return [self.progress_output]