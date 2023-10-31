import numpy as np
import pytest
import torch
from bonner.computation.decomposition._cca import CCAPytorch


def create_random_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = 1000

    rng = np.random.default_rng()

    latvar1 = rng.normal(size=n_samples).astype(float)
    latvar2 = rng.normal(size=n_samples).astype(float)

    indep1 = rng.normal(size=(n_samples, 4)).astype(float)
    indep2 = rng.normal(size=(n_samples, 5)).astype(float)

    # Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
    data1 = 0.25 * indep1 + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2)).T
    data2 = (
        0.25 * indep2
        + 0.75 * np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T
    )

    # Split each dataset into two halves: training set and test set
    train1 = data1[: n_samples // 2]
    train2 = data2[: n_samples // 2]
    test1 = data1[n_samples // 2 :]
    test2 = data2[n_samples // 2 :]

    return train1, train2, test1, test2


# @pytest.mark.parametrize("regularization", [0, 0.1, 1, 10])
@pytest.mark.parametrize("regularization", [0, 0.1, 1, 10])
def test_cca(regularization: float) -> None:
    train1, train2, test1, test2 = create_random_data()
    cca_rcca = rcca.CCA(kernelcca=False, reg=regularization, numCC=2)
    cca_rcca.train([train1, train2])
    test_correlations_rcca = cca_rcca.validate([test1, test2])

    cca_torch = CCAPytorch(
        alpha=regularization,
        n_components=2,
        kwargs_kernel=None,
        device="cpu",
    )
    cca_torch.fit([torch.from_numpy(train1), torch.from_numpy(train2)])
    test_correlations_torch = cca_torch.validate(
        [torch.from_numpy(test1), torch.from_numpy(test2)],
    )
    for _rcca, _torch in zip(test_correlations_rcca, test_correlations_torch):
        assert np.allclose(_rcca, _torch.cpu().numpy(), rtol=0.01)
    assert np.allclose(
        cca_rcca.cancorrs,
        cca_torch._canonical_correlations.cpu().numpy(),
        rtol=0.01,
    )
    print(1)
