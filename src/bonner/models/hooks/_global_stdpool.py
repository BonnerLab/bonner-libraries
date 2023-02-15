import torch

from bonner.models.hooks._definition import Hook


class GlobalStdpool(Hook):
    def __init__(self) -> None:
        super().__init__(identifier="global_stdpool")

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        match features.ndim:
            case 4:  # normal conv stdpool
                return features.std(dim=[-2, -1], keepdim=False)
            case 3:  # stdpool across patches in ViT
                return features.std(dim=1)
            case _:
                raise ValueError("features do not have the appropriate shape")
