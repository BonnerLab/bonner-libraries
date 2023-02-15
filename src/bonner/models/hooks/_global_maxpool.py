import torch

from bonner.models.hooks._definition import Hook


class GlobalMaxpool(Hook):
    def __init__(self) -> None:
        super().__init__(identifier="global_maxpool")

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Globally max-pools the features along all spatial dimensions.

        WARNING: this function assumes that

            * all 4-D features are from convolutional layers and have the shape ``(presentation, channel, spatial_x, spatial_y)``
            * all 3-D features are from patch-based Vision Transformers and have the shape ``(presentation, patch, channel)``

        Args:
            features: features extracted from a convolutional layer (with shape ``(presentation, channel, spatial_x, spatial_y)``) or from a patch-based Vision Transformer (with shape ``(presentation, patch, channel)``)

        Returns:
            spatially max-pooled features
        """
        match features.ndim:
            case 4:  # normal conv maxpool
                return features.amax(dim=[-2, -1])
            case 3:  # maxpool across patches in ViT
                return features.amax(dim=1)
            case _:
                raise ValueError("features do not have the appropriate shape")
