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
        normalize: bool = False,
        truncated: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_samples: int
        self.seed = seed
        self.center = center
        self.normalize = normalize
        self.truncated = truncated

        self.left_mean: torch.Tensor
        self.right_mean: torch.Tensor
        self.left_std: torch.Tensor
        self.right_std: torch.Tensor
        self.left_dimensions_included: Sequence[bool]
        self.right_dimensions_included: Sequence[bool]
        self.singular_values: torch.Tensor
        self.left_singular_vectors: torch.Tensor
        self.right_singular_vectors: torch.Tensor

        self.device: torch.device

    def to(self, device: torch.device | str) -> None:
        self.left_mean = self.left_mean.to(device)
        self.right_mean = self.right_mean.to(device)
        self.left_std = self.left_std.to(device)
        self.right_std = self.right_std.to(device)
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

        if x.shape[-2] != y.shape[-2]:
            raise ValueError("x and y must have the same number of samples")
        self.n_samples = x.shape[-2]

        if x.dtype != y.dtype:
            raise ValueError("x and y must have the same dtype")
        if x.device != y.device:
            raise ValueError("x and y must be on the same device")
        self.device = x.device

        x = torch.clone(x)
        y = torch.clone(y)

        # normalize data (divide each dimension by its standard deviation)
        # if std == 0, then remove those dimensions from analysis
        if self.normalize:
            self.left_std = x.std(dim=-2, keepdim=True)
            self.left_dimensions_included = (self.left_std != 0).squeeze()
            self.left_std = self.left_std[..., self.left_dimensions_included]
            x = x[..., self.left_dimensions_included]
            x /= self.left_std

            self.right_std = y.std(dim=-2, keepdim=True)
            self.right_dimensions_included = (self.right_std != 0).squeeze()
            self.right_std = self.right_std[..., self.right_dimensions_included]
            y = y[..., self.right_dimensions_included]
            y /= self.right_std
        else:
            self.left_std = torch.ones(1, device=self.device)
            self.right_std = torch.ones(1, device=self.device)
            self.left_dimensions_included = [True] * x.shape[-1]
            self.right_dimensions_included = [True] * y.shape[-1]

        # check for max_n_components here because we might be removing dimensions during normalization
        max_n_components = min(self.n_samples, x.shape[-1], y.shape[-1])
        if self.n_components is None:
            self.n_components = max_n_components
        else:
            if self.n_components > max_n_components:
                raise ValueError(f"n_components must be <= {max_n_components}")

        if self.center:
            self.left_mean = x.mean(dim=-2, keepdim=True)
            x -= self.left_mean

            self.right_mean = y.mean(dim=-2, keepdim=True)
            y -= self.right_mean
        else:
            self.left_mean = torch.zeros(1, device=self.device)
            self.right_mean = torch.zeros(1, device=self.device)

        return x, y

    def fit(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = self._preprocess(x=x, y=y)

        if self.truncated:
            torch.manual_seed(self.seed)
            u, s, v = torch.pca_lowrank(
                x.transpose(-2, -1) @ y, center=False, q=self.n_components
            )
            v_h = v.transpose(-2, -1)
        else:
            u, s, v_h = torch.linalg.svd(x.transpose(-2, -1) @ y, full_matrices=False)
        u, v_h = svd_flip(u=u, v=v_h)

        self.left_singular_vectors = u[..., : self.n_components]
        self.right_singular_vectors = v_h[..., : self.n_components, :].transpose(-2, -1)
        self.singular_values = s[..., : self.n_components] / (self.n_samples - 1)

    def transform(self, z: torch.Tensor, *, direction: str) -> torch.Tensor:
        match direction:
            case "left":
                mean = self.left_mean
                std = self.left_std
                projection = self.left_singular_vectors
                dimensions_included = self.left_dimensions_included
            case "right":
                mean = self.right_mean
                std = self.right_std
                projection = self.right_singular_vectors
                dimensions_included = self.right_dimensions_included
            case _:
                raise ValueError("direction must be 'left' or 'right'")

        z = torch.clone(z)
        z = z.to(self.device)
        z = z[..., dimensions_included]
        z = (z - mean) / std
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
                std = self.left_std
            case "right":
                projection = self.right_singular_vectors
                mean = self.right_mean
                std = self.right_std
            case _:
                raise ValueError("direction must be 'left' or 'right'")
        # TODO add back excluded dimensions as NaNs
        return (z @ projection[..., :, components].transpose(-2, -1)) * std + mean
