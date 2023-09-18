from collections.abc import Sequence

import torch
from bonner.computation.decomposition._svd import svd


class PLSSVD:
    def __init__(
        self,
        *,
        n_components: int | None = None,
        seed: int = 0,
        center: bool = True,
        scale: bool = False,
        truncated: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_samples: int
        self.seed = seed
        self.center = center
        self.scale = scale
        self.truncated = truncated

        self.left_mean: torch.Tensor
        self.right_mean: torch.Tensor
        self.left_std: torch.Tensor
        self.right_std: torch.Tensor
        self.singular_values: torch.Tensor
        self.left_singular_vectors: torch.Tensor
        self.right_singular_vectors: torch.Tensor

        self.device: torch.device

    def to(self, device: torch.device) -> None:
        self.left_mean = self.left_mean.to(device)
        self.right_mean = self.right_mean.to(device)
        self.left_std = self.left_std.to(device)
        self.right_std = self.right_std.to(device)
        self.left_singular_vectors = self.left_singular_vectors.to(device)
        self.right_singular_vectors = self.right_singular_vectors.to(device)
        self.singular_values = self.singular_values.to(device)

        self.device = torch.device(device)

    def _preprocess(
        self, x: torch.Tensor, y: torch.Tensor, /,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        if y.ndim == 1:
            y = y.unsqueeze(dim=-1)

        if x.shape[-2] != y.shape[-2]:
            raise ValueError("x and y must have the same number of samples")
        self.n_samples = x.shape[-2]

        max_n_components = min(self.n_samples, x.shape[-1], y.shape[-1])
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

        # scale data (divide each dimension by its standard deviation)
        if self.scale:
            self.left_std = x.std(dim=-2, keepdim=True)
            self.left_std[self.left_std == 0] = 1
            x /= self.left_std

            self.right_std = y.std(dim=-2, keepdim=True)
            self.right_std[self.right_std == 0] = 1
            y /= self.right_std
        else:
            self.left_std = torch.ones(1, device=self.device)
            self.right_std = torch.ones(1, device=self.device)

        return x, y

    def fit(self, x: torch.Tensor, y: torch.Tensor, /) -> None:
        x, y = self._preprocess(x, y)

        if torch.equal(x, y):
            _, s, v_h = svd(
                x,
                truncated=self.truncated,
                n_components=self.n_components,
                seed=self.seed,
            )
            u = v_h.transpose(-2, -1)
            s = s**2
        else:
            u, s, v_h = svd(
                x.transpose(-2, -1) @ y,
                truncated=self.truncated,
                n_components=self.n_components,
                seed=self.seed,
            )

        self.left_singular_vectors = u[..., : self.n_components]
        self.right_singular_vectors = v_h[..., : self.n_components, :].transpose(-2, -1)
        self.singular_values = s[..., : self.n_components] / (self.n_samples - 1)

    def transform(self, z: torch.Tensor, /, *, direction: str) -> torch.Tensor:
        match direction:
            case "left":
                mean = self.left_mean
                std = self.left_std
                projection = self.left_singular_vectors
            case "right":
                mean = self.right_mean
                std = self.right_std
                projection = self.right_singular_vectors
            case _:
                raise ValueError("direction must be 'left' or 'right'")

        z = torch.clone(z)
        z = z.to(self.device)
        z = (z - mean) / std
        return z @ projection

    def inverse_transform(
        self,
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
                std = self.left_std
            case "right":
                projection = self.right_singular_vectors
                mean = self.right_mean
                std = self.right_std
            case _:
                raise ValueError("direction must be 'left' or 'right'")
        return (z @ projection[..., :, components].transpose(-2, -1)) * std + mean
