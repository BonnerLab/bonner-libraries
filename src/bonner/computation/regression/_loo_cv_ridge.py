"""Ridge regression with generalized cross-validation (GCV) for alpha selection.

GPU-native, supports per-target alpha selection. Picks between a primal (SVD-on-X)
and dual (eigen-on-XX^T) GCV path based on shape, matching sklearn's
``RidgeCV(gcv_mode='auto')``. Scoring uses leave-one-out MSE (sklearn default).
"""

from collections.abc import Collection
from typing import Literal, Self

import numpy as np
import torch
from loguru import logger

from bonner.computation.regression._definition import Regression


class RidgeGCV(Regression):
    def __init__(
        self: Self,
        l2_penalties: Collection[float | int] | None = None,
        *,
        fit_intercept: bool = True,
        scale_x: bool = False,
        alpha_per_target: bool = True,
        gcv_mode: Literal["auto", "svd", "eigen"] = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Fit ridge regression, choosing the penalty by generalized cross-validation.

        Every candidate penalty is scored from a single decomposition of the design, so the sweep
        costs one decomposition rather than one fit per penalty. The leave-one-out errors are
        obtained in closed form rather than by refitting.

        Basic usage:

        ```
        model = RidgeGCV(l2_penalties=[1.0, 10.0, 100.0])
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)
        chosen = model.alpha_
        ```

        After ``fit``, ``alpha_`` holds the selected penalty — one per target when
        ``alpha_per_target`` is set — and ``loo_errors`` holds the scores the selection maximised.
        Those scores are *negated* mean squared errors, so larger is better despite the name.

        Args:
        ----
            l2_penalties: candidate penalties; defaults to a log-spaced grid spanning 1e-3 to 1e4
            fit_intercept: centre the design and targets, then recover the intercept
            scale_x: divide each feature by its standard deviation before fitting
            alpha_per_target: choose a penalty independently for each target rather than one
                penalty shared across all of them
            gcv_mode: which decomposition to score from — ``"svd"`` of the design, ``"eigen"`` of
                the Gram matrix, or ``"auto"``, which takes the cheaper one for the input's shape
            device: device to fit on
            dtype: cast inputs to this dtype before fitting; ``None`` keeps theirs

        """
        if l2_penalties is None:
            l2_penalties = np.logspace(-3, 4, 8).tolist()
        self.l2_penalties = l2_penalties
        self.fit_intercept = fit_intercept
        self.scale_x = scale_x
        self.alpha_per_target = alpha_per_target
        self.gcv_mode = gcv_mode
        self.device = device
        self.dtype = dtype

        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None
        self.alpha_: torch.Tensor | None = None
        self.loo_errors: torch.Tensor | None = None
        self._mode_logged = False

    def to(self: Self, device: torch.device | str) -> None:
        """Move the fitted coefficients and intercept to a device, and fit there in future.

        Args:
        ----
            device: destination device

        """
        self.device = device
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)

    def weights(self: Self) -> torch.Tensor:
        """Return the fitted coefficients, or ``None`` if the model has not been fitted.

        Returns:
        -------
            coefficients (n_features, n_targets)

        """
        return self.coefficients

    @staticmethod
    def _decomp_diag(v_prime: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        return (v_prime * Q**2).sum(axis=-1)

    @staticmethod
    def _diag_dot(D: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B

    @staticmethod
    def _find_smallest_angle(query: torch.Tensor, vectors: torch.Tensor) -> int:
        abs_cosine = torch.abs(torch.matmul(query, vectors))
        return torch.argmax(abs_cosine).item()

    def _resolve_mode(self: Self, n_samples: int, n_features: int) -> str:
        if self.gcv_mode == "auto":
            return "svd" if n_samples > n_features else "eigen"
        return self.gcv_mode

    def _eigen_decompose_gram(
        self: Self, x: torch.Tensor, y: torch.Tensor, sqrt_sw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x_mean = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        K = x @ x.T
        if self.fit_intercept:
            K += torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)
        QT_y = Q.T @ y
        return x_mean, eigvals, Q, QT_y

    def _solve_eigen_gram(
        self: Self,
        alpha: float,
        y: torch.Tensor,
        sqrt_sw: torch.Tensor,
        x_mean: torch.Tensor,
        eigvals: torch.Tensor,
        Q: torch.Tensor,
        QT_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / torch.linalg.norm(sqrt_sw)
            intercept_dim = self._find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0
        c = Q @ self._diag_dot(w, QT_y)
        G_inverse_diag = self._decomp_diag(w, Q)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c

    def _svd_decompose_design_matrix(
        self: Self, x: torch.Tensor, y: torch.Tensor, sqrt_sw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # ``U``/``singvals_sq``/``UT_y`` stay in float64 — the downstream
        # cancellation ``w = (s²+α)⁻¹ − α⁻¹`` mis-ranks adjacent alphas in float32.
        x_mean = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        x_aug = torch.hstack([x, sqrt_sw[:, None]]) if self.fit_intercept else x
        U, singvals, _ = torch.linalg.svd(x_aug.to(torch.float64), full_matrices=False)
        singvals_sq = singvals**2
        UT_y = U.T @ y.to(torch.float64)
        return x_mean, singvals_sq, U, UT_y

    def _solve_svd_design_matrix(
        self: Self,
        alpha: float,
        y: torch.Tensor,
        sqrt_sw: torch.Tensor,
        x_mean: torch.Tensor,
        singvals_sq: torch.Tensor,
        U: torch.Tensor,
        UT_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = (singvals_sq + alpha) ** -1 - alpha**-1
        if self.fit_intercept:
            normalized_sw = (sqrt_sw / torch.linalg.norm(sqrt_sw)).to(U.dtype)
            intercept_dim = self._find_smallest_angle(normalized_sw, U)
            w[intercept_dim] = -(alpha**-1)
        c = U @ self._diag_dot(w, UT_y) + (alpha**-1) * y.to(U.dtype)
        G_inverse_diag = self._decomp_diag(w, U) + (alpha**-1)
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c

    def fit(self: Self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the model, selecting the penalty and storing the coefficients and intercept.

        Also populates ``alpha_`` and ``loo_errors``. A one-dimensional ``y`` is treated as a
        single target.

        Args:
        ----
            x: predictors (n_samples, n_features)
            y: targets (n_samples, n_targets) or (n_samples,)

        """
        x = x.to(self.device)
        y = y.to(self.device)
        if self.dtype is not None:
            x = x.to(self.dtype)
            y = y.to(self.dtype)
        y_was_1d = y.ndim == 1
        if y_was_1d:
            y = y.unsqueeze(-1)

        x_offset = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        y_offset = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
        x_scale = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)

        if self.fit_intercept:
            x_offset = x.mean(dim=0)
            y_offset = y.mean(dim=0)
            x = x - x_offset
            y = y - y_offset
        if self.scale_x:
            x_scale = x.std(dim=0, correction=1)
            x_scale[x_scale == 0.0] = 1.0
            x = x / x_scale

        alphas = torch.as_tensor(list(self.l2_penalties), dtype=torch.float32)

        mode = self._resolve_mode(x.shape[0], x.shape[1])
        if not self._mode_logged:
            logger.info(
                "RidgeGCV: gcv_mode={!r} resolved to {!r} (n_samples={}, n_features={})",
                self.gcv_mode, mode, x.shape[0], x.shape[1],
            )
            self._mode_logged = True
        if mode == "svd":
            decompose, solve = self._svd_decompose_design_matrix, self._solve_svd_design_matrix
        else:
            decompose, solve = self._eigen_decompose_gram, self._solve_eigen_gram

        sqrt_sw = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        x_mean, *decomposition = decompose(x, y, sqrt_sw)

        n_y = 1 if y.ndim == 1 else y.shape[1]
        best_alpha, best_coef, best_score = None, None, None
        loo_errors = []

        for alpha in torch.atleast_1d(alphas):
            G_inverse_diag, coef = solve(
                float(alpha), y, sqrt_sw, x_mean, *decomposition,
            )
            squared_errors = (coef / G_inverse_diag) ** 2
            if self.alpha_per_target:
                score = -squared_errors.mean(dim=0)
            else:
                score = -squared_errors.mean()
            loo_errors.append(score.detach())

            if best_score is None:
                best_alpha = alpha
                best_coef = coef
                best_score = score
                if self.alpha_per_target and n_y > 1:
                    best_alpha = torch.full((n_y,), alpha)
            else:
                if self.alpha_per_target and n_y > 1:
                    to_update = score > best_score
                    best_alpha[to_update] = alpha
                    best_coef[:, to_update] = coef[:, to_update]
                    best_score[to_update] = score[to_update]
                elif score > best_score:
                    best_alpha, best_coef, best_score = alpha, coef, score

        self.alpha_ = best_alpha
        self.loo_errors = torch.stack(loo_errors)
        dual_coef = best_coef.to(x.dtype) if best_coef.dtype != x.dtype else best_coef
        self.coefficients = (dual_coef.T @ x).T  # shape (n_features, n_targets)
        x_offset = x_offset + x_mean * x_scale
        if self.fit_intercept:
            self.coefficients = self.coefficients / x_scale[:, None]
            self.intercept = y_offset - x_offset @ self.coefficients
        else:
            self.intercept = torch.zeros(1, dtype=x.dtype, device=x.device)

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets for new predictors, moving them to the coefficients' device.

        Args:
        ----
            x: predictors (n_samples, n_features)

        Returns:
        -------
            predictions (n_samples, n_targets)

        """
        x = x.to(self.coefficients.device)
        if self.dtype is not None:
            x = x.to(self.dtype)
        return x @ self.coefficients + self.intercept


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, p = 2000, 128
    x = torch.randn(n, p)
    w = torch.randn(p, p) * 0.1
    y = x @ w + 0.01 * torch.randn(n, p)
    ridge = RidgeGCV(device=device)
    ridge.fit(x, y)
    pred = ridge.predict(x.to(device))
    ratio = (torch.linalg.norm(y.to(device) - pred) / torch.linalg.norm(y.to(device))).item()
    logger.info("smoke test: ||resid||/||target|| = {:.4f}", ratio)
    assert ratio < 0.05, f"smoke test failed: ratio={ratio:.4f}"
    logger.info("smoke test PASSED")
