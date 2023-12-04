from collections.abc import Collection
from typing import Self

import torch
from bonner.computation.regression._utilities import Regression


def columnwise_mean_square(x: torch.Tensor) -> torch.Tensor:
    return (x**2).mean(dim=-2)


class RidgeGCV(Regression):
    def __init__(
        self: Self,
        l2_penalties: Collection[float | int],
        fit_intercept: bool = True,
    ) -> None:
        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

        self.fit_intercept = fit_intercept
        self.l2_penalties = l2_penalties

        self.loo_errors: torch.Tensor | None = None

    def to(self: Self, device: torch.device | str) -> None:
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)

    def fit(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        x = torch.clone(x)
        y = torch.clone(y).to(x.device)

        x = x.unsqueeze(dim=-1) if x.ndim == 1 else x
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        # many sets of predictors, only 1 set of targets
        if x.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(0)

        n_samples = x.shape[-2]

        if y.shape[-2] != n_samples:
            raise ValueError(
                f"number of samples in x and y must be equal (x={n_samples},"
                f" y={y.shape[-2]})",
            )

        if self.fit_intercept:
            x_mean = x.mean(dim=-2, keepdim=True)
            x -= x_mean
            y_mean = y.mean(dim=-2, keepdim=True)
            y -= y_mean

        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        del vt, x

        identity = torch.eye(n_samples)

        self.loo_errors = torch.full(
            fill_value=torch.nan,
            size=(y.shape[-1], len(self.l2_penalties)),
        )

        for i_penalty, l2_penalty in enumerate(self.l2_penalties):
            s_bar = torch.diag(-(s**2) / (l2_penalty * s**2 + l2_penalty**2))
            a = u @ s_bar @ u.T + identity / l2_penalty

            self.loo_errors[:, i_penalty] = columnwise_mean_square(
                a @ y / torch.diag(a),
            )

        raise NotImplementedError
        self.coefficients = vt.transpose(-2, -1) @ (d * (u.transpose(-2, -1) @ y))

        if self.fit_intercept:
            self.intercept = y_mean - x_mean @ self.coefficients
        else:
            self.intercept = torch.zeros(1)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.coefficients.device) @ self.coefficients + self.intercept
