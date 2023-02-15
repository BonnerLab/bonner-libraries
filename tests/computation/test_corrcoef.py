from typing import Callable
import pytest

import numpy as np
import torch

from bonner.computation.metrics import covariance, pearson_r


def _create_data(
    x_shape: tuple[int, ...], y_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed=0)

    x_numpy = rng.normal(size=x_shape)
    y_numpy = rng.normal(size=y_shape)

    x_torch = torch.from_numpy(x_numpy.copy())
    y_torch = torch.from_numpy(y_numpy.copy())

    return x_numpy, y_numpy, x_torch, y_torch


def _get_functions(metric: str) -> tuple[Callable, Callable]:
    match metric:
        case "correlation":
            fn_numpy = np.corrcoef
            fn_torch = pearson_r
        case "covariance":
            fn_numpy = np.cov
            fn_torch = covariance
    return fn_numpy, fn_torch


@pytest.mark.parametrize("metric", ["correlation", "covariance"])
def test_corrcoef_1d_1d(
    metric: str,
    n_samples: int = 5,
) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples,),
        y_shape=(n_samples,),
    )
    fn_numpy, fn_torch = _get_functions(metric)
    r_numpy = fn_numpy(x_numpy, y_numpy)[0, 1]
    r_torch = fn_torch(x_torch, y_torch, return_diagonal=True).cpu().numpy()

    assert np.allclose(r_numpy, r_torch)


@pytest.mark.parametrize("metric", ["correlation", "covariance"])
def test_pearson_r_1d_2d(metric: str, n_samples: int = 5, n_features: int = 3) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples,),
        y_shape=(n_samples, n_features),
    )
    fn_numpy, fn_torch = _get_functions(metric)

    r_numpy = np.empty(n_features)
    for i_feature in range(n_features):
        r_numpy[i_feature] = fn_numpy(x_numpy, y_numpy[:, i_feature], rowvar=False)[
            0, 1
        ]
    r_torch = fn_torch(x_torch, y_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy, r_torch)

    r_torch = fn_torch(y_torch, x_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy, r_torch)


def test_pearson_r_1d_3d(
    n_batch: int = 10, n_samples: int = 5, n_features: int = 3
) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples,),
        y_shape=(n_batch, n_samples, n_features),
    )

    r_numpy = np.empty((n_batch, n_features))
    for i_batch in range(n_batch):
        for i_feature in range(n_features):
            r_numpy[i_batch, i_feature] = np.corrcoef(
                x_numpy, y_numpy[i_batch, :, i_feature], rowvar=False
            )[0, 1]
    r_torch = pearson_r(x_torch, y_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy, r_torch)

    r_torch = pearson_r(y_torch, x_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy, r_torch)


@pytest.mark.parametrize("n_features_x,n_features_y", [(3, 4), (4, 4)])
def test_pearson_r_2d_2d(
    n_features_x: int,
    n_features_y: int,
    n_samples: int = 5,
) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples, n_features_x),
        y_shape=(n_samples, n_features_y),
    )

    r_numpy = np.corrcoef(x_numpy, y_numpy, rowvar=False)[:n_features_x, n_features_x:]
    r_torch = pearson_r(x_torch, y_torch, return_diagonal=False).cpu().numpy()

    assert np.allclose(r_numpy, r_torch)

    if n_features_x == n_features_y:
        r_numpy = np.diag(r_numpy)
        r_torch = pearson_r(x_torch, y_torch, return_diagonal=True).cpu().numpy()
        assert np.allclose(r_numpy, r_torch)


@pytest.mark.parametrize("n_features_x,n_features_y", [(3, 4), (4, 4)])
def test_pearson_r_2d_3d(
    n_features_x: int,
    n_features_y: int,
    n_batch: int = 10,
    n_samples: int = 5,
) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples, n_features_x),
        y_shape=(n_batch, n_samples, n_features_y),
    )

    r_numpy_xy = np.empty((n_batch, n_features_x, n_features_y))
    r_numpy_yx = np.empty((n_batch, n_features_y, n_features_x))
    for i_batch in range(n_batch):
        r_numpy_xy[i_batch, ...] = np.corrcoef(
            x_numpy, y_numpy[i_batch, ...], rowvar=False
        )[:n_features_x, n_features_x:]
        r_numpy_yx[i_batch, ...] = np.corrcoef(
            y_numpy[i_batch, ...], x_numpy, rowvar=False
        )[:n_features_y, n_features_y:]
    r_torch_xy = pearson_r(x_torch, y_torch, return_diagonal=False).cpu().numpy()
    r_torch_yx = pearson_r(y_torch, x_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy_xy, r_torch_xy)
    assert np.allclose(r_numpy_yx, r_torch_yx)
    assert np.allclose(r_torch_xy, r_torch_yx.transpose((0, 2, 1)))

    if n_features_x == n_features_y:
        r_numpy_xy = np.diagonal(r_numpy_xy, axis1=-2, axis2=-1)
        r_torch_xy = pearson_r(x_torch, y_torch, return_diagonal=True).cpu().numpy()
        assert np.allclose(r_numpy_xy, r_torch_xy)

        r_numpy_yx = np.diagonal(r_numpy_yx, axis1=-2, axis2=-1)
        r_torch_yx = pearson_r(y_torch, x_torch, return_diagonal=True).cpu().numpy()
        assert np.allclose(r_numpy_yx, r_torch_yx)

        assert np.allclose(r_torch_xy, r_torch_yx)


@pytest.mark.parametrize("n_features_x,n_features_y", [(3, 4), (4, 4)])
def test_pearson_r_3d_3d(
    n_features_x: int,
    n_features_y: int,
    n_batch: int = 10,
    n_samples: int = 5,
) -> None:
    x_numpy, y_numpy, x_torch, y_torch = _create_data(
        x_shape=(n_samples, n_features_x),
        y_shape=(n_batch, n_samples, n_features_y),
    )

    r_numpy = np.empty((n_batch, n_features_x, n_features_y))
    for i_batch in range(n_batch):
        r_numpy[i_batch, ...] = np.corrcoef(
            x_numpy, y_numpy[i_batch, ...], rowvar=False
        )[:n_features_x, n_features_x:]
    r_torch = pearson_r(x_torch, y_torch, return_diagonal=False).cpu().numpy()
    assert np.allclose(r_numpy, r_torch)

    if n_features_x == n_features_y:
        r_numpy = np.diagonal(r_numpy, axis1=-2, axis2=-1)
        r_torch = pearson_r(x_torch, y_torch, return_diagonal=True).cpu().numpy()
        assert np.allclose(r_numpy, r_torch)
