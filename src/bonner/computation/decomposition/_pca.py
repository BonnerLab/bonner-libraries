from collections.abc import Sequence

import torch


class PCA:
    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components

        self.mean: torch.Tensor
        self.eigenvectors: torch.Tensor
        self.eigenvalues: torch.Tensor

    def to(self, device: torch.device | str) -> None:
        self.mean = self.mean.to(device)
        self.eigenvectors = self.eigenvectors.to(device)
        self.eigenvalues = self.eigenvalues.to(device)

    def fit(self, x: torch.Tensor) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        n_samples, n_features = x.shape[-2], x.shape[-1]
        max_n_components = min(n_samples, n_features)

        if self.n_components is None:
            self.n_components = max_n_components
        else:
            if self.n_components > max_n_components:
                raise ValueError(f"n_components must be <= {max_n_components}")

        x = torch.clone(x)
        self.mean = x.mean(dim=-2, keepdim=True)
        x -= self.mean

        u, s, v_h = torch.linalg.svd(x, full_matrices=False)
        u, v_h = _svd_flip(u=u, v=v_h)

        self.eigenvectors = v_h[..., : self.n_components, :].transpose(-2, -1)
        self.eigenvalues = (s[..., : self.n_components] ** 2) / (n_samples - 1)

    def transform(self, x: torch.Tensor, *, whiten: bool = True) -> torch.Tensor:
        x = torch.clone(x)
        x = x.to(self.mean.device)
        x -= self.mean

        nonzero = self.eigenvalues != 0

        x_transformed = torch.matmul(x, self.eigenvectors[..., :, nonzero])
        if whiten:
            x_transformed /= torch.sqrt(self.eigenvalues[..., nonzero].unsqueeze(-2))
        return x_transformed

    def inverse_transform(
        self,
        x: torch.Tensor,
        *,
        components: Sequence[int] | int = None,
        whiten: bool = False,
    ) -> torch.Tensor:
        if components is None:
            components = self.n_components
        if isinstance(components, int):
            components = [_ for _ in range(components)]

        if whiten:
            return (
                torch.matmul(
                    x[..., components],
                    torch.sqrt(self.eigenvalues[..., components].unsqueeze(-1))
                    * self.eigenvectors[..., :, components].transpose(-2, -1),
                )
                + self.mean
            )
        else:
            return (
                torch.matmul(
                    x[..., components],
                    self.eigenvectors[..., :, components].transpose(-2, -1),
                )
                + self.mean
            )


def _svd_flip(*, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs_cols = torch.argmax(torch.abs(u), dim=-2)
    match u.ndim:
        case 3:
            signs = torch.stack(
                [
                    torch.sign(u[i_batch, max_abs_cols[i_batch, :], range(u.shape[-1])])
                    for i_batch in range(u.shape[0])
                ],
                dim=0,
            )
        case 2:
            signs = torch.sign(u[..., max_abs_cols, range(u.shape[-1])])

    u *= signs.unsqueeze(-2)
    v *= signs.unsqueeze(-1)

    return u, v
