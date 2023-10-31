from typing import Self

import torch
from bonner.models.hooks._definition import Hook


class GlobalMaxpool(Hook):
    def __init__(
        self: Self,
        amax_dim: int | list[int] | None = None,
        even: bool | None = None,
        band_width: int = 1,
    ) -> None:
        identifier = "global_maxpool"
        if amax_dim:
            identifier += f".amax_dim={amax_dim}".replace(" ", "")
            if isinstance(amax_dim, int):
                amax_dim = [amax_dim]
        if even is not None:
            identifier += f".even={even}.band_width={band_width}"
        super().__init__(identifier=identifier)

        self.amax_dim = amax_dim
        self.even = even
        self.band_width = band_width

    def __call__(self: Self, features: torch.Tensor) -> torch.Tensor:
        """Globally max-pools the features along all spatial dimensions.

        WARNING: this function assumes that

            * all 4-D features are from convolutional layers and have the shape ``(presentation, channel, spatial_x, spatial_y)``
            * UPDATE: exception of 4-D features: VITs and convnext from torchvision have shape (presentation, *, *, channel)
            * all 3-D features are from patch-based Vision Transformers and have the shape ``(presentation, patch, channel)``

        Args:
        ----
            features: features extracted from a convolutional layer (with shape ``(presentation, channel, spatial_x, spatial_y)``) or from a patch-based Vision Transformer (with shape ``(presentation, patch, channel)``)

        Returns:
        -------
            spatially max-pooled features
        """
        if self.even is None:
            # for edge cases noted above
            if self.amax_dim:
                assert features.ndim == 4
                return features.amax(dim=self.amax_dim)

            match features.ndim:
                case 4:  # normal conv maxpool
                    return features.amax(dim=[-2, -1])
                case 3:  # maxpool across patches in ViT
                    return features.amax(dim=1)
                case _:
                    raise ValueError("features do not have the appropriate shape")
        else:
            start_idx = 0 if self.even else self.band_width
            if self.amax_dim:
                assert features.ndim == 4
                end_idx = features.size(self.amax_dim[0])
            else:
                match features.ndim:
                    case 4:  # normal conv maxpool
                        end_idx = features.size(-2)
                    case 3:  # maxpool across patches in ViT
                        end_idx = features.size(1)
                    case _:
                        raise ValueError("features do not have the appropriate shape")

            band_count = (end_idx - start_idx) // (2 * self.band_width)
            selected_indices = [
                i
                for b in range(band_count)
                for i in range(
                    start_idx + b * 2 * self.band_width,
                    start_idx + b * 2 * self.band_width + self.band_width,
                )
            ]

            remaining_start = band_count * 2 * self.band_width + start_idx
            selected_indices.extend(
                range(remaining_start, min(remaining_start + self.band_width, end_idx)),
            )

            if self.amax_dim:
                for ad in self.amax_dim:
                    features = features.index_select(ad, torch.tensor(selected_indices))
                return features.amax(dim=self.amax_dim)

            match features.ndim:
                case 4:  # normal conv maxpool
                    features[..., selected_indices][..., selected_indices, :]
                    return features.amax(dim=[-2, -1])
                case 3:  # maxpool across patches in ViT
                    features = features[:, selected_indices, :]
                    return features.amax(dim=1)
