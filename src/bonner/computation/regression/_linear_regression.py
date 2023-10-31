from typing import Self

import torch
from bonner.computation.regression._utilities import Regression

EPSILON = 1e-15


class LinearRegression(Regression):
    def __init__(
        self: Self,
        *,
        fit_intercept: bool = True,
        l2_penalty: float | torch.Tensor | None = None,
        rcond: float | None = None,
        driver: str | None = None,
        allow_ols_on_cuda: bool = False,
    ) -> None:
        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

        self.fit_intercept = fit_intercept
        self.l2_penalty = l2_penalty
        self.rcond = rcond
        self.driver = driver
        self.allow_ols_on_cuda = allow_ols_on_cuda

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

        n_samples, n_features = x.shape[-2], x.shape[-1]

        # TODO: underdetermined systems on CUDA use a different driver
        if (
            (not self.allow_ols_on_cuda)
            and (self.l2_penalty is None)
            and (n_samples < n_features)
        ):
            x = x.to(torch.device("cpu"))
            y = y.to(torch.device("cpu"))

        if y.shape[-2] != n_samples:
            error = (
                f"number of samples in x and y must be equal (x={n_samples},"
                f" y={y.shape[-2]})",
            )
            raise ValueError(error)

        if self.fit_intercept:
            x_mean = x.mean(dim=-2, keepdim=True)
            x -= x_mean
            y_mean = y.mean(dim=-2, keepdim=True)
            y -= y_mean

        if self.l2_penalty is None:
            self.coefficients, _, _, _ = torch.linalg.lstsq(
                x,
                y,
                rcond=self.rcond,
                driver=self.driver,
            )
        else:
            if isinstance(self.l2_penalty, float | int) or (
                isinstance(self.l2_penalty, torch.Tensor)
                and self.l2_penalty.numel() == 1
            ):
                l2_penalty = self.l2_penalty * torch.ones(y.shape[-1], device=x.device)
            elif isinstance(self.l2_penalty, torch.Tensor):
                l2_penalty = self.l2_penalty.to(x.device)

            u, s, vt = torch.linalg.svd(x, full_matrices=False)
            idx = s > EPSILON
            s_nnz = s[idx].unsqueeze(-1)
            d = torch.zeros(
                size=(len(s), l2_penalty.numel()),
                dtype=x.dtype,
                device=x.device,
            )
            d[idx] = s_nnz / (s_nnz**2 + l2_penalty)
            self.coefficients = vt.transpose(-2, -1) @ (d * (u.transpose(-2, -1) @ y))

        if self.fit_intercept:
            self.intercept = y_mean - x_mean @ self.coefficients
        else:
            self.intercept = torch.zeros(1)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.coefficients.device) @ self.coefficients + self.intercept
