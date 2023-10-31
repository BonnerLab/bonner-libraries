from typing import Self

import torch
from bonner.models.hooks._definition import Hook


class Flatten(Hook):
    def __init__(self: Self) -> None:
        super().__init__(identifier="flatten")

    def __call__(self: Self, features: torch.Tensor, /) -> torch.Tensor:
        """Flattens the features along all spatial dimensions.

        Args:
        ----
            features: features extracted from a node (with shape ``(presentation, ...)``)

        Returns:
        -------
            flattened features
        """
        return torch.flatten(features, start_dim=1)
