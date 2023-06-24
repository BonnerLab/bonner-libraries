from collections.abc import Sequence

import torch
from bonner.computation.decomposition._utilities import svd_flip


class PCA:
    def __init__(
        self, 
        n_components: int = None,
        scale: bool = False,
        truncated: bool = False,
        seed: int = 0
    ) -> None:
        self.n_components = n_components
        self.n_samples: int
        self.scale = scale
        self.truncated = truncated
        self.seed = seed

        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.eigenvectors: torch.Tensor
        self.eigenvalues: torch.Tensor

        self.device: torch.device

    def to(self, device: torch.device | str) -> None:
        self.mean = self.mean.to(device)
        self.eigenvectors = self.eigenvectors.to(device)
        self.eigenvalues = self.eigenvalues.to(device)

        self.device = torch.device(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        self.n_samples, n_features = x.shape[-2], x.shape[-1]
        max_n_components = min(self.n_samples, n_features)

        if self.n_components is None:
            self.n_components = max_n_components
        else:
            if self.n_components > max_n_components:
                raise ValueError(f"n_components must be <= {max_n_components}")
        self.device = x.device

        return x

    def fit(self, x: torch.Tensor) -> None:
        x = self._preprocess(x)

        x = torch.clone(x)
        self.mean = x.mean(dim=-2, keepdim=True)
        x -= self.mean
        
        if self.scale:
            self.std = x.std(dim=-2, keepdim=True)
            self.std[self.std == 0] = 1
            x /= self.std
        else:
            self.std = torch.ones(1, device=self.device)
            
        if self.truncated:
            torch.manual_seed(self.seed)
            u, s, v = torch.pca_lowrank(x, center=False, q=self.n_components)
            v_h = v.transpose(-2, -1)
        else:
            u, s, v_h = torch.linalg.svd(x, full_matrices=False)
        u, v_h = svd_flip(u=u, v=v_h)

        self.eigenvectors = v_h[..., : self.n_components, :].transpose(-2, -1)
        self.eigenvalues = (s[..., : self.n_components] ** 2) / (self.n_samples - 1)

    def transform(
        self,
        z: torch.Tensor,
        *,
        components: Sequence[int] | int = None,
    ) -> torch.Tensor:
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = torch.clone(z)
        z = z.to(self.device)
        z -= self.mean
        z /= self.std

        return z @ self.eigenvectors[..., components]

    def inverse_transform(
        self,
        z: torch.Tensor,
        *,
        components: Sequence[int] | int = None,
    ) -> torch.Tensor:
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)
        z = z[..., components]

        return (z @ self.eigenvectors[..., components].transpose(-2, -1)) * self.std + self.mean
