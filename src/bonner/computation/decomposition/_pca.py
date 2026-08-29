from collections.abc import Sequence
from typing import Self

import torch
from bonner.computation.decomposition._svd import svd


class PCA:
    def __init__(
        self: Self,
        *,
        n_components: int | None = None,
        scale: bool = False,
        randomized: bool = False,
        truncated: bool = False,
        seed: int = 0,
    ) -> None:
        """Principal component analysis by singular value decomposition.

        The data are centred, and optionally standardized, before decomposition. The eigenvalues
        are ``s ** 2 / (n_samples - 1)`` for singular values ``s``, which is the scaling that makes
        them eigenvalues of the sample covariance matrix — or of the correlation matrix when
        ``scale`` is set.

        Two approximate solvers are available and they are not the same knob. ``truncated`` takes
        precedence and calls ``torch.pca_lowrank`` directly, reseeding the global torch RNG with
        ``seed`` so that its random projection is reproducible. ``randomized`` reaches the same
        solver through ``svd`` and seeds nothing, so ``seed`` has no effect unless ``truncated``
        is set.

        Basic usage:

        ```
        pca = PCA(n_components=10)
        pca.fit(x_train)
        scores = pca.transform(x_test)
        ```

        Args:
        ----
            n_components: number of components to keep; ``None`` keeps
                ``min(n_samples, n_features)``, and ``fit`` writes the resolved value back to this
                attribute
            scale: divide each feature by its standard deviation before decomposing
            randomized: use the approximate solver reached through ``svd``
            truncated: use ``torch.pca_lowrank`` directly, seeded from ``seed``
            seed: seed for the ``truncated`` solver only

        """
        self.n_components = n_components
        self.n_samples: int
        self.scale = scale
        self.randomized = randomized
        self.truncated = truncated
        self.seed = seed

        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.eigenvectors: torch.Tensor
        self.eigenvalues: torch.Tensor

        self.device: torch.device

    def to(self: Self, device: torch.device | str) -> None:
        """Move the fitted statistics to a device.

        The standard deviations are not moved. A model fitted on one device and moved to another
        therefore raises a device mismatch on the next ``transform`` or ``inverse_transform``,
        whether or not ``scale`` was set; move the fitted array yourself if you need this.

        Args:
        ----
            device: destination device

        """
        self.mean = self.mean.to(device)
        self.eigenvectors = self.eigenvectors.to(device)
        self.eigenvalues = self.eigenvalues.to(device)

        self.device = torch.device(device)

    def _preprocess(self: Self, x: torch.Tensor, /) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        self.n_samples, n_features = x.shape[-2], x.shape[-1]
        max_n_components = min(self.n_samples, n_features)

        if self.n_components is None:
            self.n_components = max_n_components
        elif self.n_components > max_n_components:
            error = f"n_components must be <= {max_n_components}"
            raise ValueError(error)
        self.device = x.device

        x = torch.clone(x)
        self.mean = x.mean(dim=-2, keepdim=True)
        x -= self.mean

        if self.scale:
            self.std = x.std(dim=-2, keepdim=True)
            self.std[self.std == 0] = 1
            x /= self.std
        else:
            self.std = torch.ones(1, device=self.device)

        return x

    def fit(self: Self, x: torch.Tensor, /) -> None:
        """Fit the decomposition, storing the centring statistics and the eigendecomposition.

        Also resolves ``n_components`` when it was ``None``, and pins the model to ``x``'s device.
        A one-dimensional ``x`` is read as a single feature.

        Args:
        ----
            x: data to fit (*, n_samples, n_features)

        """
        x = self._preprocess(x)

        if self.truncated:
            from bonner.computation.decomposition._svd import _svd_flip
            torch.manual_seed(self.seed)
            u, s, v = torch.pca_lowrank(x, center=False, q=self.n_components)
            v_h = v.transpose(-2, -1)
            u, v_h = _svd_flip(u=u, v_h=v_h)
            self.eigenvectors = v_h[..., : self.n_components, :].transpose(-2, -1)
            del u, v, v_h
        else:
            _, s, self.eigenvectors = svd(
                x,
                randomized=self.randomized,
                n_components=self.n_components,
            )
            del _

        self.eigenvalues = (s[..., : self.n_components] ** 2) / (self.n_samples - 1)

    def transform(
        self: Self,
        z: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Project data onto the fitted components.

        ``z`` is moved to the model's device, then centred and scaled with the statistics from
        ``fit`` — not with its own — so it may be data the model never saw.

        Args:
        ----
            z: data to project (*, n_samples, n_features)
            components: components to project onto; an integer is read as the leading that many,
                and ``None`` uses all of them

        Returns:
        -------
            scores (*, n_samples, n_selected_components)

        """
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
        self: Self,
        z: torch.Tensor,
        /,
        *,
        components: Sequence[int] | int | None = None,
    ) -> torch.Tensor:
        """Map scores back to the original feature space, undoing the centring and scaling.

        ``components`` selects columns of ``z`` as well as components of the basis, so ``z`` must
        be in the full component space. This is not the inverse of ``transform`` called with the
        same argument: ``transform`` has already dropped the unselected columns, so round-tripping
        a non-leading selection such as ``[2, 3]`` raises an ``IndexError``.

        Args:
        ----
            z: scores in the full component space (*, n_samples, n_components)
            components: components to reconstruct from; an integer is read as the leading that
                many, and ``None`` uses all of them

        Returns:
        -------
            reconstructed data (*, n_samples, n_features)

        """
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = list(range(components))

        z = z.to(self.device)
        z = z[..., components]

        return (
            z @ self.eigenvectors[..., components].transpose(-2, -1)
        ) * self.std + self.mean
