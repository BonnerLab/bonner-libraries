from collections.abc import Sequence
from typing import Self

import torch

from bonner.computation.decomposition._svd import svd


class PLSSVD:
    def __init__(
        self: Self,
        *,
        randomized: bool,
    ) -> None:
        self.randomized = randomized

        self.n_samples: int
        self.n_components: int

        self.left_mean: torch.Tensor
        self.right_mean: torch.Tensor
        self.singular_values: torch.Tensor
        self.left_singular_vectors: torch.Tensor
        self.right_singular_vectors: torch.Tensor

        self.device: torch.device

    def to(self: Self, device: torch.device) -> None:
        self.left_mean = self.left_mean.to(device)
        self.right_mean = self.right_mean.to(device)
        self.left_singular_vectors = self.left_singular_vectors.to(device)
        self.right_singular_vectors = self.right_singular_vectors.to(device)
        self.singular_values = self.singular_values.to(device)

        self.device = torch.device(device)

    def _preprocess(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        if y.ndim == 1:
            y = y.unsqueeze(dim=-1)

        if x.shape[-2] != y.shape[-2]:
            error = "x and y must have the same number of samples"
            raise ValueError(error)
        self.n_samples = x.shape[-2]

        if x.dtype != y.dtype:
            error = "x and y must have the same dtype"
            raise ValueError(error)
        if x.device != y.device:
            error = "x and y must be on the same device"
            raise ValueError(error)
        self.device = x.device

        x = torch.clone(x)
        y = torch.clone(y)

        # center the data
        self.left_mean = x.mean(dim=-2, keepdim=True)
        x -= self.left_mean

        self.right_mean = y.mean(dim=-2, keepdim=True)
        y -= self.right_mean

        self.n_components = min(
            x.shape[-2],
            x.shape[-1],
            y.shape[-1],
        )

        return x, y

    def fit(self: Self, x: torch.Tensor, y: torch.Tensor, /) -> None:
        x, y = self._preprocess(x, y)

        if torch.equal(x, y):
            _, s, v = svd(
                x,
                randomized=self.randomized,
                n_components=self.n_components,
            )
            u = v
            s = s**2
        else:
            u, s, v = svd(
                x.transpose(-2, -1) @ y,
                randomized=self.randomized,
                n_components=self.n_components,
            )

        self.left_singular_vectors = u
        self.right_singular_vectors = v
        self.singular_values = s / (self.n_samples - 1)

    def transform(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        direction: str,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        match direction:
            case "left":
                mean = self.left_mean
                projection = self.left_singular_vectors
            case "right":
                mean = self.right_mean
                projection = self.right_singular_vectors
            case _:
                error = "direction must be 'left' or 'right'"
                raise ValueError(error)

        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = torch.clone(z)
        z = z.to(self.device)
        z = z - mean
        return z @ projection[..., components]

    def inverse_transform(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        direction: str,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)
        z = z[..., components]

        match direction:
            case "left":
                projection = self.left_singular_vectors
                mean = self.left_mean
            case "right":
                projection = self.right_singular_vectors
                mean = self.right_mean
            case _:
                error = "direction must be 'left' or 'right'"
                raise ValueError(error)
        return (z @ projection[..., components].transpose(-2, -1)) + mean
