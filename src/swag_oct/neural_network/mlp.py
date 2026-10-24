"""A small multilayer perceptron that returns one binary-classification logit."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 128), dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("Размеры слоёв должны быть положительными")
        if not 0 <= dropout < 1:
            raise ValueError("dropout должен лежать в [0, 1)")

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)
