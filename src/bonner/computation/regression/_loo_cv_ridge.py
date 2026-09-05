"""Ridge regression whose penalty is chosen per target by closed-form leave-one-out error.

The penalty can be named in either of two ways, and which one is right depends on what is
being compared.

**By L2 penalty** (``l2_penalties``), the ordinary parameterization. One grid of penalties is
tried and each target keeps the one with the lowest leave-one-out error.

**By fraction** (``fractions``), the reparameterization of Rokem & Kay: instead of a penalty,
name the ratio ``gamma = ||b_ridge|| / ||b_ols||`` between the regularized and unregularized
coefficient norms, and solve for the penalty that achieves it. This matters whenever fits over
designs of different width or scale are compared against each other. A penalty is not
comparable between such designs -- the same number is a different amount of shrinkage
depending on the design's singular values -- so a fixed penalty grid applies unequal
regularization across a comparison and can run out at one end for some designs and not others,
which is a bias in the comparison rather than noise in it. A fraction is scale-free and
bounded to ``(0, 1]``, so one grid spans the whole achievable range of shrinkage for every
design.

    model = RidgeGCV(fractions=[0.05, 0.1, 0.5, 1.0])
    model.fit(x=x_train, y=y_train)
    model.predict(x_test)
    model.fraction_            # the fraction each target selected
    model.achieved_fractions_  # what those selections actually realize

``achieved_fractions_`` is worth reading rather than assuming: the penalty for a requested
fraction is found by interpolating a curve of realized fractions, so a requested value is a
target and not a guarantee.

Both routes share one decomposition of the design, one selection rule, and one solve, and the
selection is vectorized over per-target penalties -- which is what the fractional route needs,
since two targets asking for the same fraction generally need different penalties.

The leave-one-out error is taken in the hat-matrix form, ``(y - yhat) / (1 - H_jj)`` with
``H_jj`` summed from the shrinkage factors. The algebraically equivalent kernel form needs a
term in ``1 / penalty``, which is a subtraction of two nearly equal large numbers at a small
penalty and undefined at zero -- and zero is where the fractional route's unregularized end
lands.

An intercept enters as an extra unpenalized direction of the design rather than by centering
alone. Centering the design and fitting without an intercept makes the leave-one-out error an
approximation, because holding a sample out also moves the means it was centred by: the error
that introduces falls off as one over the sample count, which is small but is a bias in the
selection rather than noise in it. Carrying the intercept as a column of the design instead
lets the closed form account for that shift, and makes the leave-one-out error exact.
"""

from collections.abc import Collection, Sequence
from typing import Self

import numpy as np
import torch
from loguru import logger

from bonner.computation.regression._definition import Regression

#: Penalties tried when neither grid is named.
DEFAULT_L2_PENALTIES: tuple[float, ...] = tuple(np.logspace(-3, 4, 8).tolist())

#: Fractions from the paper the fractional route follows: linearly spaced, unregularized end
#: included. A caller comparing designs whose targets are mostly poorly predicted will want a
#: grid that resolves heavy shrinkage more finely, since those targets pile up near zero.
DEFAULT_FRACTIONS: tuple[float, ...] = tuple(np.linspace(0.05, 1.0, 20).tolist())

#: Spacing, in log10 units, of the internal penalty grid whose realized fractions are
#: interpolated. Finer than the paper's, which leaves the smallest fractions several percent
#: from what was asked; the grid enters only the norm-ratio curve and not the selection loop,
#: so refining it is close to free. ``achieved_fractions_`` is what reports the residual.
_SEARCH_STEP = 0.05

#: How far the internal grid extends past the design's squared singular values at either end,
#: in log10 units. The penalty achieving any fraction in ``(0, 1]`` is guaranteed to lie
#: inside the range this produces.
_SEARCH_MARGIN = 3.0


class RidgeGCV(Regression):
    def __init__(
        self: Self,
        l2_penalties: Collection[float | int] | None = None,
        *,
        fractions: Collection[float] | None = None,
        fit_intercept: bool = True,
        scale_x: bool = False,
        alpha_per_target: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        """Fit ridge regression, choosing the penalty by leave-one-out cross-validation.

        Every candidate is scored from a single decomposition of the design, so the sweep
        costs one decomposition rather than one fit per candidate, and the leave-one-out
        errors are obtained in closed form rather than by refitting.

        Args:
        ----
            l2_penalties: candidate penalties. Mutually exclusive with ``fractions``;
                defaults to a log-spaced grid when neither is given
            fractions: candidate coefficient-norm ratios in ``(0, 1]``, selecting the
                fractional parameterization
            fit_intercept: centre the design and targets, then recover the intercept
            scale_x: divide each predictor by its standard deviation before fitting
            alpha_per_target: choose independently for each target rather than once for all.
                Ignored on the fractional route, where the penalty is per target by
                construction even when every target requests the same fraction
            device: device to fit on
            dtype: cast inputs to this dtype before fitting; ``None`` keeps theirs

        """
        if l2_penalties is not None and fractions is not None:
            msg = "name either l2_penalties or fractions, not both"
            raise ValueError(msg)

        self.fractions = None if fractions is None else tuple(float(f) for f in fractions)
        if self.fractions is not None:
            if not all(0.0 < f <= 1.0 for f in self.fractions):
                msg = f"fractions must lie in (0, 1]; got {self.fractions}"
                raise ValueError(msg)
            self.l2_penalties: tuple[float, ...] = ()
        else:
            self.l2_penalties = tuple(
                float(p) for p in (DEFAULT_L2_PENALTIES if l2_penalties is None else l2_penalties)
            )
            if any(p < 0 for p in self.l2_penalties):
                msg = f"penalties must be non-negative; got {self.l2_penalties}"
                raise ValueError(msg)

        self.fit_intercept = fit_intercept
        self.scale_x = scale_x
        self.alpha_per_target = alpha_per_target
        self.device: torch.device | str = device
        self.dtype = dtype

        self.coefficients: torch.Tensor | None = None
        self.intercept: torch.Tensor | None = None
        self.alpha_: torch.Tensor | None = None
        self.fraction_: torch.Tensor | None = None
        self.achieved_fractions_: torch.Tensor | None = None
        self.loo_errors: torch.Tensor | None = None

    def to(self: Self, device: torch.device | str) -> None:
        """Move the fitted coefficients and intercept to a device, and fit there in future."""
        self.device = device
        if self.coefficients is not None:
            self.coefficients = self.coefficients.to(device)
        if self.intercept is not None:
            self.intercept = self.intercept.to(device)

    def weights(self: Self) -> torch.Tensor | None:
        """The fitted coefficients ``(n_features, n_targets)``, or ``None`` before fitting."""
        return self.coefficients

    @staticmethod
    def _rank_filter(singular_values: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Which singular values are treated as non-zero.

        The threshold is the one ``torch.linalg.pinv`` applies, so a design this call
        considers rank-deficient is one the library's own pseudo-inverse would too. Directions
        below it carry no information and would otherwise divide the unregularized solution by
        a value indistinguishable from zero.
        """
        tolerance = (
            singular_values.max()
            * max(n_samples, singular_values.numel())
            * torch.finfo(singular_values.dtype).eps
        )
        return singular_values > tolerance

    @staticmethod
    def _interpolate(
        query: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Piecewise-linear interpolation of ``y`` at ``query``, per column of ``x``.

        ``x`` is ``(grid, targets)`` and must ascend down each column; ``y`` is ``(grid,)``
        and is shared. Queries outside a column's range take that column's end value, which is
        the correct saturation here: past either end the realized fraction has stopped moving,
        so the penalty that achieves it is the endpoint's.
        """
        upper = torch.searchsorted(x.T.contiguous(), query.expand(x.shape[1], -1).contiguous())
        upper = upper.clamp(1, x.shape[0] - 1)
        lower = upper - 1
        columns = torch.arange(x.shape[1], device=x.device)[:, None]
        x_low, x_high = x[lower, columns], x[upper, columns]
        y_low, y_high = y[lower], y[upper]
        weight = (query[None, :] - x_low) / (x_high - x_low).clamp_min(
            torch.finfo(x.dtype).tiny
        )
        interpolated = y_low + weight.clamp(0.0, 1.0) * (y_high - y_low)
        return interpolated.T

    @staticmethod
    def _intercept_direction(u: torch.Tensor) -> int:
        """Which left singular vector is the intercept column of an augmented design.

        The design is centred before the column of ones is appended, so that column is
        orthogonal to every other and appears as one exact singular direction. Located by
        alignment rather than by position, since the decomposition orders by singular value
        and the intercept's is not distinguished by size.
        """
        ones = torch.full(
            (u.shape[0],), u.shape[0] ** -0.5, dtype=u.dtype, device=u.device
        )
        return int(torch.argmax(torch.abs(ones @ u)))

    def _penalties_for_fractions(
        self: Self, singular_values: torch.Tensor, ut_y: torch.Tensor
    ) -> torch.Tensor:
        """Penalties realizing each requested fraction, one per (fraction, target).

        Realized fractions are evaluated on an internal penalty grid spanning the design's
        squared singular values with a margin at either end, then inverted by interpolation.
        The curve is monotone decreasing in the penalty, so it is reversed to ascend before
        interpolating, and the interpolation is done in ``log1p`` of the penalty because the
        grid is log-spaced and spans many orders of magnitude.
        """
        squared = singular_values**2
        exponents = torch.arange(
            float(torch.log10(squared.min()).item()) - _SEARCH_MARGIN,
            float(torch.log10(squared.max()).item()) + _SEARCH_MARGIN,
            _SEARCH_STEP,
            device=squared.device,
            dtype=squared.dtype,
        )
        grid = torch.cat([torch.zeros(1, device=squared.device, dtype=squared.dtype),
                          10.0**exponents])

        shrinkage = squared[None, :] / (squared[None, :] + grid[:, None])
        unregularized = ut_y / singular_values[:, None]
        norms = torch.sqrt(shrinkage**2 @ unregularized**2)
        realized = norms / norms[0:1, :].clamp_min(torch.finfo(norms.dtype).tiny)

        requested = torch.as_tensor(
            self.fractions, device=squared.device, dtype=squared.dtype
        )
        log_penalties = self._interpolate(
            requested, torch.flip(realized, dims=[0]), torch.flip(torch.log1p(grid), dims=[0])
        )
        return torch.expm1(log_penalties)

    def _score(
        self: Self,
        shrinkage: torch.Tensor,
        u: torch.Tensor,
        ut_y: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Negated mean squared leave-one-out error per target, for one candidate.

        ``shrinkage`` is ``(rank, targets)``. Negated so that larger is better, matching how
        the selection below reads it.
        """
        predicted = u @ (shrinkage * ut_y)
        leverage = (u**2) @ shrinkage
        residual = (y - predicted) / (1.0 - leverage).clamp_min(
            torch.finfo(y.dtype).eps
        )
        return -(residual**2).mean(dim=0)

    def fit(self: Self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the model, selecting per target and storing the coefficients and intercept.

        Args:
        ----
            x: predictors ``(n_samples, n_features)``
            y: targets ``(n_samples, n_targets)`` or ``(n_samples,)``

        """
        x = x.to(self.device)
        y = y.to(self.device)
        if self.dtype is not None:
            x, y = x.to(self.dtype), y.to(self.dtype)
        target_was_1d = y.ndim == 1
        if target_was_1d:
            y = y.unsqueeze(-1)

        x_offset = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        y_offset = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
        x_scale = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)
        if self.fit_intercept:
            x_offset, y_offset = x.mean(dim=0), y.mean(dim=0)
            x, y = x - x_offset, y - y_offset
        if self.scale_x:
            x_scale = x.std(dim=0, correction=1)
            x_scale[x_scale == 0.0] = 1.0
            x = x / x_scale

        design = (
            torch.hstack([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)])
            if self.fit_intercept
            else x
        )
        u, singular_values, vt = torch.linalg.svd(design, full_matrices=False)
        keep = self._rank_filter(singular_values, design.shape[0])
        u, singular_values, vt = u[:, keep], singular_values[keep], vt[keep, :]
        squared = singular_values**2
        ut_y = u.T @ y
        n_targets = y.shape[1]

        # The intercept direction is exempt from the penalty: shrinking it would regularize
        # the mean. It is left out of the coefficient norm the fractional route takes its
        # ratio over as well, which is definitional rather than load-bearing -- the targets
        # are centred, so that direction's unregularized coefficient is already zero to
        # rounding, and including it would change no ratio here. It would matter to a caller
        # who reached the same code with uncentred targets.
        penalized = torch.ones(singular_values.numel(), dtype=torch.bool, device=x.device)
        intercept_direction = -1
        if self.fit_intercept:
            intercept_direction = self._intercept_direction(u)
            penalized[intercept_direction] = False

        if self.fractions is not None:
            penalties = self._penalties_for_fractions(
                singular_values[penalized], ut_y[penalized]
            )
        else:
            penalties = torch.as_tensor(
                self.l2_penalties, device=x.device, dtype=x.dtype
            )[:, None].expand(-1, n_targets)

        best_score = torch.full((n_targets,), -float("inf"), dtype=y.dtype, device=x.device)
        best_shrinkage = torch.zeros(
            singular_values.numel(), n_targets, dtype=x.dtype, device=x.device
        )
        best_index = torch.zeros(n_targets, dtype=torch.long, device=x.device)
        scores = []
        for index in range(penalties.shape[0]):
            shrinkage = squared[:, None] / (squared[:, None] + penalties[index][None, :])
            if intercept_direction >= 0:
                shrinkage[intercept_direction] = 1.0
            score = self._score(shrinkage, u, ut_y, y)
            scores.append(score)
            if self.alpha_per_target or self.fractions is not None:
                better = score > best_score
            else:
                # One candidate for every target, so the comparison is between whole-grid
                # means and its outcome is the same for each: expanded rather than per target.
                better = (score.mean() > best_score.mean()).expand(n_targets)
            best_score = torch.where(better, score, best_score)
            best_shrinkage = torch.where(better[None, :], shrinkage, best_shrinkage)
            best_index = torch.where(better, index, best_index)

        self.loo_errors = torch.stack(scores)
        self.alpha_ = penalties[best_index, torch.arange(n_targets, device=x.device)]
        if self.fractions is not None:
            requested = torch.as_tensor(self.fractions, device=x.device, dtype=x.dtype)
            self.fraction_ = requested[best_index]

        unregularized = ut_y / singular_values[:, None]
        rotated = best_shrinkage * unregularized
        if self.fractions is not None:
            self.achieved_fractions_ = torch.linalg.norm(
                rotated[penalized], dim=0
            ) / torch.linalg.norm(unregularized[penalized], dim=0).clamp_min(
                torch.finfo(x.dtype).tiny
            )

        augmented = vt.T @ rotated
        self.coefficients = augmented[: x.shape[1]]
        if self.scale_x:
            self.coefficients = self.coefficients / x_scale[:, None]
        # The centred problem's own intercept is the augmented column's coefficient. It is
        # not zero: the unpenalized direction absorbs whatever the shrinkage of the others
        # leaves behind, which is exactly what makes the leave-one-out error exact.
        centred_intercept = (
            augmented[x.shape[1]] if self.fit_intercept else torch.zeros(1, dtype=x.dtype)
        )
        self.intercept = (
            y_offset + centred_intercept - x_offset @ self.coefficients
            if self.fit_intercept
            else torch.zeros(n_targets, dtype=x.dtype, device=x.device)
        )
        if target_was_1d:
            logger.debug("RidgeGCV: a one-dimensional target was fitted as a single target")

    def predict(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets for new predictors, moving them to the coefficients' device.

        Args:
        ----
            x: predictors ``(n_samples, n_features)``

        Returns:
        -------
            predictions ``(n_samples, n_targets)``

        """
        if self.coefficients is None or self.intercept is None:
            msg = "the model has not been fitted"
            raise RuntimeError(msg)
        x = x.to(self.coefficients.device)
        if self.dtype is not None:
            x = x.to(self.dtype)
        return x @ self.coefficients + self.intercept


__all__ = ["DEFAULT_FRACTIONS", "DEFAULT_L2_PENALTIES", "RidgeGCV"]
