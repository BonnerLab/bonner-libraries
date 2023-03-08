from collections.abc import Sequence

import torch
from bonner.computation.decomposition._utilities import svd_flip


class SVCA:
    def __init__(
        self,
        *,
        n_components: int = None,
        seed: int = 0,
        center: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_samples: int
        self.seed = seed
        self.center = center

        self.left_mean: torch.Tensor
        self.right_mean: torch.Tensor
        self.singular_values: torch.Tensor
        self.left_singular_vectors: torch.Tensor
        self.right_singular_vectors: torch.Tensor

        self.device: torch.device

    def to(self, device: torch.device | str) -> None:
        self.left_mean = self.left_mean.to(device)
        self.right_mean = self.right_mean.to(device)
        self.left_singular_vectors = self.left_singular_vectors.to(device)
        self.right_singular_vectors = self.right_singular_vectors.to(device)
        self.singular_values = self.singular_values.to(device)

        self.device = torch.device(device)

    def _preprocess(
        self, *, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        if y.ndim == 1:
            y = y.unsqueeze(dim=-1)

        n_samples_x, n_features_x = x.shape[-2], x.shape[-1]
        n_samples_y, n_features_y = y.shape[-2], y.shape[-1]
        if n_samples_x != n_samples_y:
            raise ValueError("x and y must have the same number of samples")
        self.n_samples = n_samples_x

        max_n_components = min(self.n_samples, n_features_x, n_features_y)
        if self.n_components is None:
            self.n_components = max_n_components
        else:
            if self.n_components > max_n_components:
                raise ValueError(f"n_components must be <= {max_n_components}")

        if x.dtype != y.dtype:
            raise ValueError("x and y must have the same dtype")
        if x.device != y.device:
            raise ValueError("x and y must be on the same device")
        self.device = x.device

        return x, y

    def fit(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = self._preprocess(x=x, y=y)

        x = torch.clone(x)
        y = torch.clone(y)

        if self.center:
            self.left_mean = x.mean(dim=-2, keepdim=True)
            x -= self.left_mean

            self.right_mean = y.mean(dim=-2, keepdim=True)
            y -= self.right_mean
        else:
            self.left_mean = torch.zeros(1, device=self.device)
            self.right_mean = torch.zeros(1, device=self.device)

        torch.manual_seed(self.seed)
        u, s, v = torch.pca_lowrank(
            x.transpose(-2, -1) @ y, center=False, q=self.n_components
        )
        v_h = v.transpose(-2, -1)
        u, v_h = svd_flip(u=u, v=v_h)

        self.left_singular_vectors = u[..., : self.n_components]
        self.right_singular_vectors = v_h[..., : self.n_components, :].transpose(-2, -1)
        self.singular_values = s[..., : self.n_components] / (self.n_samples - 1)

    def transform(self, z: torch.Tensor, *, direction: str) -> torch.Tensor:
        match direction:
            case "left":
                mean = self.left_mean
                projection = self.left_singular_vectors
            case "right":
                mean = self.right_mean
                projection = self.right_singular_vectors
            case _:
                raise ValueError("direction must be 'left' or 'right'")

        z = torch.clone(z)
        z = z.to(self.device)
        z -= mean
        return z @ projection

    def inverse_transform(
        self,
        z: torch.Tensor = None,
        *,
        direction: str,
        components: Sequence[int] | int = None,
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
                raise ValueError("direction must be 'left' or 'right'")
        return (z @ projection[..., :, components].transpose(-2, -1)) + mean
