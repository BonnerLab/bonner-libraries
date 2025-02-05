from typing import Self

import numpy as np
import torch
from bonner.computation.cuda import try_devices
from bonner.models.hooks._definition import Hook


def compute_johnson_lindenstrauss_limit(*, n_samples: int, epsilon: float) -> int:
    return int(np.ceil(4 * np.log(n_samples) / ((epsilon**2) / 2 - (epsilon**3) / 3)))


def create_sparse_projection_matrix(
    *,
    n_features: int,
    n_components: int,
    density: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    assert isinstance(n_features, int), "n_features must be an int"
    assert n_features > 1, "n_features must be > 1"

    if density is None:
        density = np.exp(-np.log(n_features) / 2)
    else:
        assert isinstance(density, float)
        assert density > 0, "density must be > 0"
        assert density <= 1, "density must be <= 1"

    assert isinstance(n_components, int), "n_components must be an int"
    assert n_components >= 1, "n_components must be >= 1"

    scale = np.exp(-(np.log(density) + np.log(n_components)) / 2)

    n_elements = n_features * n_components

    rng = np.random.default_rng(seed=seed)
    n_nonzero = rng.binomial(n=n_elements, p=density, size=1)[0]
    indices = rng.choice(a=n_elements, size=n_nonzero, replace=False).astype(np.int64)
    locations = np.stack(
        np.unravel_index(indices=indices, shape=(n_features, n_components)),
    )

    return torch.sparse_coo_tensor(
        indices=torch.from_numpy(locations),
        values=scale
        * (2 * rng.binomial(n=1, p=0.5, size=n_nonzero) - 1).astype(np.float32),
        size=(n_features, n_components),
    )


class SparseRandomProjection(Hook):
    def __init__(
        self: Self,
        *,
        n_components: int,
        density: float | None = None,
        seed: int = 0,
        allow_expansion: bool = False,
    ) -> None:
        self.n_components = n_components
        self.density = density
        self.seed = seed
        self.allow_expansion = allow_expansion

        super().__init__(
            identifier=(
                "sparse_random_projection"
                f".n_components={self.n_components}"
                f".density={self.density}"
                f".seed={self.seed}"
                f".allow_expansion={self.allow_expansion}"
            ),
        )

    def __call__(self: Self, features: torch.Tensor) -> torch.Tensor:
        features = features.flatten(start_dim=1)
        n_features = features.shape[-1]

        projection = create_sparse_projection_matrix(
            n_features=n_features,
            n_components=self.n_components,
            density=self.density,
            seed=self.seed,
        )

        if (n_features <= projection.shape[-1]) and not self.allow_expansion:
            return features

        return try_devices(self._project)(features=features, projection=projection)

    def _project(
        self: Self,
        *,
        features: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        return features @ projection
