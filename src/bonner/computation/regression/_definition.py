from abc import ABC, abstractmethod
from typing import Self

import torch


class Regression(ABC):
    @abstractmethod
    def fit(self: Self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        pass

    @abstractmethod
    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def weights(self: Self) -> torch.Tensor:
        pass
