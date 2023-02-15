import pytest

import numpy as np
import torch
from sklearn.metrics import r2_score as r2_score_sklearn

from bonner.computation.metrics import r2_score


@pytest.mark.parametrize("shape", [(10,), (5, 10)])
def test_r2_score(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(seed=0)
    y_numpy = rng.normal(size=shape)
    y_numpy_predicted = rng.normal(size=shape)

    y_torch = torch.from_numpy(y_numpy.copy())
    y_torch_predicted = torch.from_numpy(y_numpy_predicted.copy())

    r2 = r2_score(y_torch, y_torch_predicted).cpu().numpy()
    r2_sklearn = r2_score_sklearn(y_numpy, y_numpy_predicted, multioutput="raw_values")
    assert np.allclose(r2, r2_sklearn)
