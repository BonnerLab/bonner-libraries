import logging
from typing import Self

import torch
from bonner.computation.regression._utilities import Regression


def _get_first_singular_vectors_power_method(
    x: torch.Tensor,
    y: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(x.dtype).eps
    y_score = next(col for col in y.T if torch.any(torch.abs(col) > eps))
    x_weights_old = 100
    for _ in range(max_iter):
        x_weights = (x.T @ y_score) / (y_score @ y_score)
        x_weights /= torch.sqrt(x_weights @ x_weights) + eps
        x_score = x @ x_weights
        y_weights = (y.T @ x_score) / (x_score @ x_score)
        y_score = (y @ y_weights) / ((y_weights @ y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if (x_weights_diff @ x_weights_diff) < tol or y.shape[1] == 1:
            break
        x_weights_old = x_weights
    return x_weights, y_weights


def _svd_flip_1d(u: torch.Tensor, v: torch.Tensor) -> None:
    biggest_abs_val_idx = torch.argmax(torch.abs(u))
    sign = torch.sign(u[biggest_abs_val_idx])
    # in-place operation
    u *= sign
    v *= sign


# only supports 2D tensors
class PLSRegression(Regression):
    def __init__(
        self: Self,
        n_components: int = 25,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def fit(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        x = torch.clone(x).to(self.device)
        y = torch.clone(y).to(x.device)

        x = x.unsqueeze(dim=-1) if x.ndim == 1 else x
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        assert x.ndim == 2 and y.ndim == 2, "Only 2D tensors are supported now"

        N, P = x.shape[-2], x.shape[-1]
        Q = y.shape[-1]

        if y.shape[-2] != N:
            error = (
                f"number of samples in x and y must be equal (x={N},"
                f" y={y.shape[-2]})",
            )
            raise ValueError(error)

        x_mean = x.mean(dim=-2, keepdim=True)
        x -= x_mean
        y_mean = y.mean(dim=-2, keepdim=True)
        y -= y_mean
        if self.scale:
            x_std = x.std(dim=-2, keepdim=True)
            y_std = y.std(dim=-2, keepdim=True)
            x_std[x_std == 0] = 1.0
            y_std[y_std == 0] = 1.0
            x /= x_std
            y /= y_std
        else:
            x_std = torch.ones(1, P, device=x.device)
            y_std = torch.ones(1, Q, device=x.device)

        x_weights_ = torch.zeros(P, self.n_components, device=x.device)  # U
        y_weights_ = torch.zeros(Q, self.n_components, device=x.device)  # V
        _x_scores = torch.zeros(N, self.n_components, device=x.device)  # Xi
        _y_scores = torch.zeros(N, self.n_components, device=x.device)  # Omega
        x_loadings_ = torch.zeros(P, self.n_components, device=x.device)  # Gamma
        y_loadings_ = torch.zeros(Q, self.n_components, device=x.device)  # Delta
        n_iter_ = []
        y_eps = torch.finfo(y.dtype).eps

        for k in range(self.n_components):
            y[:, torch.all(torch.abs(y) < 10 * y_eps, axis=0)] = 0.0

            try:
                x_weights, y_weights = _get_first_singular_vectors_power_method(
                    x,
                    y,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
            except:
                logging.info(f"Y residual is constant at iteration {k}")
                x_weights_ = x_weights_[..., : k - 1]
                y_weights_ = y_weights_[..., : k - 1]
                _x_scores = _x_scores[..., : k - 1]
                _y_scores = _y_scores[..., : k - 1]
                x_loadings_ = x_loadings_[..., : k - 1]
                y_loadings_ = y_loadings_[..., : k - 1]
                break
            _svd_flip_1d(x_weights, y_weights)
            x_scores = x @ x_weights
            y_scores = (y @ y_weights) / (y_weights @ y_weights)

            # Deflation: subtract rank-one approx to obtain Xk+1 and Yk+1
            x_loadings = (x_scores @ x) / (x_scores @ x_scores)
            x -= torch.outer(x_scores, x_loadings)
            y_loadings = (x_scores @ y) / (x_scores @ x_scores)
            y -= torch.outer(x_scores, y_loadings)

            x_weights_[..., k] = x_weights
            y_weights_[..., k] = y_weights
            _x_scores[..., k] = x_scores
            _y_scores[..., k] = y_scores
            x_loadings_[..., k] = x_loadings
            y_loadings_[..., k] = y_loadings

        x_rotations_ = x_weights_ @ torch.linalg.pinv(
            x_loadings_.T @ x_weights_,
        )
        y_rotations_ = y_weights_ @ torch.linalg.pinv(
            y_loadings_.T @ y_weights_,
        )
        self.coefficients = x_rotations_ @ y_loadings_.T
        self.coefficients = self.coefficients / x_std.T * y_std
        self.intercept = y_mean - x_mean @ self.coefficients

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.coefficients.device) @ self.coefficients + self.intercept

    def weights(self) -> torch.Tensor:
        return self.coefficients