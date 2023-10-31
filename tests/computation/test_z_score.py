import numpy as np
import pytest
import torch
from bonner.computation.normalization import z_score
from scipy.stats import zscore


@pytest.mark.parametrize("dim", [0, -1, 2])
@pytest.mark.parametrize("correction", [1, 0])
def test_z_score(
    correction: float,
    dim: int,
    shape: tuple[int, ...] = (5, 10, 3),
) -> None:
    rng = np.random.default_rng(seed=0)
    x_numpy = rng.normal(size=shape)
    x_torch = torch.from_numpy(x_numpy.copy())

    z_numpy = zscore(x_numpy, axis=dim, ddof=correction)
    z_torch = z_score(x_torch, dim=dim, correction=correction).cpu().numpy()
    assert np.allclose(z_numpy, z_torch)
