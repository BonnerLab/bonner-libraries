import numpy as np
import torch

from bonner.models.hooks._definition import Hook


class RandomProjection(Hook):
    def __init__(self, *, out_channels: int, seed: int = 0) -> None:
        self.seed = seed
        self.out_channels = out_channels
        super().__init__(
            identifier=f"random_projection.out_channels={out_channels}.seed={seed}"
        )

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Applies a random orthonormal projection to the features."""
        features = features.flatten(start_dim=1)
        in_channels = features.shape[-1]
        rng = np.random.default_rng(seed=self.seed)
        projection, _ = torch.linalg.qr(
            torch.from_numpy(rng.standard_normal(size=(in_channels, self.out_channels)))
        )
        return features @ projection
