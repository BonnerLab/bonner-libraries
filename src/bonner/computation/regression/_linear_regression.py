import math
from typing import Self

import torch
from loguru import logger

from bonner.computation.regression._utilities import Regression

EPSILON = 1e-15


class LinearRegression(Regression):
    def __init__(
        self: Self,
        *,
        fit_intercept: bool = True,
        l2_penalty: float | torch.Tensor | None = None,
        l1_penalty: float | None = None,
        l1_max_iter: int = 1000,
        l1_tol: float = 1e-7,
        rcond: float | None = None,
        driver: str | None = None,
        allow_ols_on_cuda: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if l1_penalty is not None and l2_penalty is not None:
            error = "l1_penalty and l2_penalty are mutually exclusive"
            raise ValueError(error)

        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None

        self.fit_intercept = fit_intercept
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.l1_max_iter = l1_max_iter
        self.l1_tol = l1_tol
        self.rcond = rcond
        self.driver = driver
        self.allow_ols_on_cuda = allow_ols_on_cuda
        self.device = device

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
        x = torch.clone(x).to(self.device)
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
            and (self.l1_penalty is None)
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

        if self.l1_penalty is not None:
            self.coefficients = self._fit_lasso_fista(x, y)
        elif self.l2_penalty is None:
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

    def _fit_lasso_fista(
        self: Self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """FISTA (accelerated proximal gradient) for the multi-target lasso.

        Minimises sklearn's `Lasso` objective per target, sharing one design:
            (1 / (2 * n)) * ||y - x @ w||_2^2 + alpha * ||w||_1
        `x` and `y` are already centred by `fit` (when `fit_intercept`), so the
        Lipschitz constant uses that centred design.
        """
        if x.ndim != 2:
            error = f"lasso (l1_penalty) supports only 2D x; got ndim={x.ndim}"
            raise NotImplementedError(error)

        n_samples, n_features = x.shape
        n_targets = y.shape[-1]
        alpha = float(self.l1_penalty)

        # L = sigma_max(x)^2 / n is the Lipschitz constant of grad f; step = 1/L.
        sigma_max = torch.linalg.matrix_norm(x, ord=2)
        step = (n_samples / sigma_max.pow(2)).item()
        threshold = step * alpha

        w = torch.zeros((n_features, n_targets), dtype=x.dtype, device=x.device)
        z = w.clone()
        t = 1.0

        converged = False
        rel = float("inf")
        for _ in range(self.l1_max_iter):
            w_prev = w
            grad = (x.transpose(-2, -1) @ (x @ z - y)) / n_samples
            w = self._soft_threshold(z - step * grad, threshold)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z = w + ((t - 1.0) / t_next) * (w - w_prev)
            t = t_next

            max_prev = w_prev.abs().max()
            max_change = (w - w_prev).abs().max()
            rel = (
                (max_change / max_prev).item()
                if max_prev > 0
                else max_change.item()
            )
            if rel < self.l1_tol:
                converged = True
                break

        if not converged:
            logger.warning(
                "LinearRegression lasso FISTA hit l1_max_iter={} without"
                " reaching l1_tol={} (last rel-change={:.2e})",
                self.l1_max_iter,
                self.l1_tol,
                rel,
            )
        return w

    @staticmethod
    def _soft_threshold(z: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.sign(z) * torch.clamp(z.abs() - threshold, min=0.0)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.coefficients.device) @ self.coefficients + self.intercept

    def weights(self: Self) -> torch.Tensor:
        return self.coefficients


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import Lasso

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Well-conditioned multi-target problem (n > d, low feature correlation).
    n, d, m = 400, 30, 5
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    W_true = torch.randn(d, m, generator=g, dtype=torch.float64)
    Y = X @ W_true + 0.1 * torch.randn(n, m, generator=g, dtype=torch.float64)

    alpha = 0.1
    model = LinearRegression(
        l1_penalty=alpha, l1_max_iter=50_000, l1_tol=1e-12, device=device,
    )
    model.fit(X, Y)
    ours_pred = model.predict(X.to(device)).detach().cpu().numpy()
    our_coef = model.weights().detach().cpu().numpy()

    # Reference: sklearn Lasso looped per target (unambiguous multi-output).
    X_np, Y_np = X.numpy(), Y.numpy()
    sk_pred = np.zeros_like(Y_np)
    sk_coef = np.zeros((d, m))
    for j in range(m):
        skj = Lasso(alpha=alpha, fit_intercept=True, max_iter=100_000, tol=1e-12)
        skj.fit(X_np, Y_np[:, j])
        sk_pred[:, j] = skj.predict(X_np)
        sk_coef[:, j] = skj.coef_

    max_dpred = np.abs(ours_pred - sk_pred).max()
    coef_relerr = np.abs(our_coef - sk_coef).max() / (np.abs(sk_coef).max() + 1e-12)
    print(f"lasso parity: max|dy_hat|={max_dpred:.3e}  coef relerr={coef_relerr:.3e}")
    assert max_dpred < 1e-3, f"lasso prediction parity failed: {max_dpred:.3e}"
    print("PASS: lasso FISTA matches sklearn Lasso (predictions within 1e-3)")

    # OLS / ridge paths unchanged.
    ols = LinearRegression(device=device)
    ols.fit(X, Y)
    ridge = LinearRegression(l2_penalty=1.0, device=device)
    ridge.fit(X, Y)
    print(
        "OLS/ridge fit OK; coef shapes",
        tuple(ols.weights().shape),
        tuple(ridge.weights().shape),
    )

    # Mutual-exclusion guard.
    try:
        LinearRegression(l1_penalty=0.1, l2_penalty=0.1)
    except ValueError:
        print("PASS: l1/l2 mutual-exclusion guard raises")
    else:
        raise AssertionError("expected ValueError for l1+l2")