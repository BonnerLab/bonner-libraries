from typing import Self

import numpy as np
import torch
from bonner.caching import cache
from bonner.computation.cuda import try_devices
from bonner.models.hooks._definition import Hook
from bonner.models.utilities import BONNER_MODELS_HOME


class RandomProjection(Hook):
    def __init__(
        self: Self,
        *,
        out_channels: int,
        seed: int = 0,
        allow_expansion: bool = False,
    ) -> None:
        self.seed = seed
        self.out_channels = out_channels
        self.expand = allow_expansion
        self.projections: dict[int, np.ndarray] = {}
        super().__init__(
            identifier=f"random_projection.out_channels={out_channels}.seed={seed}.expand={allow_expansion}",
        )

    def __call__(self: Self, features: torch.Tensor) -> torch.Tensor:
        """Apply a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1)
        in_channels = features.shape[-1]
        if in_channels <= self.out_channels and not self.expand:
            return features

        if in_channels not in self.projections:
            self.projections[in_channels] = self._compute_projection(
                in_channels=in_channels,
            )

        projection = torch.from_numpy(self.projections[in_channels])
        return try_devices(self._project)(features=features, projection=projection)

    def _project(
        self: Self,
        *,
        features: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        return features @ projection

    @cache(
        path=BONNER_MODELS_HOME,
        identifier=("hooks/{identifier}/in_channels={in_channels}.npy"),
        helper=lambda kwargs: {
            "identifier": kwargs["self"].identifier,
            "in_channels": kwargs["in_channels"],
        },
    )
    def _compute_projection(self: Self, *, in_channels: int) -> np.ndarray:
        rng = np.random.default_rng(seed=self.seed)
        projection, _ = torch.linalg.qr(
            torch.from_numpy(
                rng.standard_normal(
                    size=(in_channels, self.out_channels),
                    dtype=np.float32,
                ),
            ),
        )
        return projection.cpu().numpy()
