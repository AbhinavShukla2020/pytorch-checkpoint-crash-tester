from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class DeterministicRegressionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, size: int = 512, dimensions: int = 16, seed: int = 101) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(size, dimensions, generator=generator)
        weights = torch.randn(dimensions, 1, generator=generator)
        noise = 0.01 * torch.randn(size, 1, generator=generator)
        self.targets = self.features @ weights + noise

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return torch.tensor(index, dtype=torch.int64), self.features[index], self.targets[index]


class TinyRegressor(nn.Module):
    def __init__(self, dimensions: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimensions, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)
