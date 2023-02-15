import pytest

from scipy.stats import zscore
import numpy as np
import torch

from bonner.computation.normalization import z_score


@pytest.mark.parametrize("dim", [0, -1, 2])
@pytest.mark.parametrize("unbiased", [True, False])
def test_z_score(unbiased: bool, dim: int, shape: tuple[int, ...] = (5, 10, 3)) -> None:
    rng = np.random.default_rng(seed=0)
    x_numpy = rng.normal(size=shape)
    x_torch = torch.from_numpy(x_numpy.copy())

    ddof = 1 if unbiased else 0
    z_numpy = zscore(x_numpy, axis=dim, ddof=ddof)
    z_torch = z_score(x_torch, dim=dim, unbiased=unbiased).cpu().numpy()
    assert np.allclose(z_numpy, z_torch)
