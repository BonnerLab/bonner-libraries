import torch


class Hook:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        pass
