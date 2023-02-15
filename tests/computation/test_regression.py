import pytest

import numpy as np
import torch
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn, Ridge

from bonner.computation.regression import LinearRegression


class NumpyRegression:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_ = x.copy()
        y_ = y.copy()

        if self.fit_intercept:
            x_mean = x_.mean(axis=-2, keepdims=True)
            x_ = x_ - x_mean
            y_mean = y_.mean(axis=-2, keepdims=True)
            y_ = y_ - y_mean

        self.coef_, _, _, _ = np.linalg.lstsq(x_, y_, rcond=None)
        self.coef_ = self.coef_.T

        if self.fit_intercept:
            self.intercept_ = y_mean - x_mean @ self.coef_.T

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit")
        if self.fit_intercept:
            return x @ self.coef_.T + self.intercept_
        else:
            return x @ self.coef_.T


# TODO: ridge regression works identically on cuda and cpu, OLS is identical between numpy and torch-cpu, OLS is different in torch-cuda from sklearn (which uses scipy's incorrect defaults) and numpy
# @pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("l2_penalty", [None, 0, 0.1, 1, 10])
@pytest.mark.parametrize("n_samples", [5, 50])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression(
    n_features: int,
    n_targets: int,
    fit_intercept: bool,
    n_samples: int,
    seed: int,
    l2_penalty: float | None,
    device: str = "cpu",
) -> None:
    rng = np.random.default_rng(seed=seed)

    x_sklearn = rng.normal(size=(n_samples, n_features))
    beta = rng.normal(size=(n_features, n_targets))
    epsilon = 0.05 * rng.normal(size=(n_samples, n_targets))
    y_sklearn = x_sklearn @ beta + epsilon

    x_torch = torch.from_numpy(x_sklearn).to(torch.device(device))
    y_torch = torch.from_numpy(y_sklearn).to(torch.device(device))

    model_torch = LinearRegression()
    model_torch.fit(
        x=x_torch,
        y=y_torch,
        fit_intercept=fit_intercept,
        l2_penalty=l2_penalty,
    )

    if l2_penalty is None or l2_penalty == 0:
        # model_sklearn = LinearRegressionSklearn(fit_intercept=fit_intercept)
        # model_sklearn.fit(x_sklearn, y_sklearn)

        model_sklearn = NumpyRegression(fit_intercept=fit_intercept)
        model_sklearn.fit(x_sklearn, y_sklearn)
    else:
        model_sklearn = Ridge(
            alpha=l2_penalty,
            fit_intercept=fit_intercept,
            solver="svd",
        )
        model_sklearn.fit(x_sklearn, y_sklearn)

    assert np.allclose(
        model_torch.coefficients.transpose(-2, -1).cpu(), model_sklearn.coef_, atol=1e-3
    )
    if fit_intercept:
        assert np.allclose(
            model_torch.intercept.cpu(), model_sklearn.intercept_, atol=1e-3
        )
    else:
        assert model_torch.intercept == 0 and model_sklearn.intercept_ == 0

    x_test_sklearn = rng.normal(size=(100, n_features))
    x_test_torch = torch.from_numpy(x_test_sklearn).to(torch.device(device))

    y_test_sklearn = model_sklearn.predict(x_test_sklearn)
    y_test_torch = model_torch.predict(x_test_torch)

    assert np.allclose(y_test_sklearn, y_test_torch.cpu(), atol=1e-3)
