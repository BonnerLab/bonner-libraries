from typing import Self

import torch


class Hook:
    def __init__(self: Self, *, identifier: str) -> None:
        self.identifier = identifier

    def __call__(self: Self, features: torch.Tensor, /) -> torch.Tensor:
        pass
