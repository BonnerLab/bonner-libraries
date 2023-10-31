import numpy as np
import pytest
import torch
from bonner.computation.decomposition._pca import PCA
from sklearn.decomposition import PCA as PCASklearn


@pytest.mark.parametrize("shape", [(100, 10), (20, 20), (10, 100), (100, 1), (2, 10)])
@pytest.mark.parametrize("n_components_fraction", [0, 0.25, 0.5, None])
def test_pca_2d(
    shape: tuple[int, ...],
    n_components_fraction: float | None,
) -> None:
    rng = np.random.default_rng(seed=0)
    x_sklearn = rng.normal(size=shape)
    x_torch = torch.from_numpy(x_sklearn.copy())

    if n_components_fraction is not None:
        n_components = int(max(np.floor(n_components_fraction * min(shape[-2:])), 1))
    else:
        n_components = min(shape[-2:])

    if shape[-2] <= shape[-1]:
        if n_components != 1:
            n_components -= 1

    pca_sklearn = PCASklearn(n_components=n_components)
    pca_torch = PCA(n_components=n_components)

    pca_sklearn.fit(x_sklearn)
    pca_torch.fit(x_torch)

    assert np.allclose(
        pca_sklearn.components_,
        pca_torch.eigenvectors.transpose(-2, -1),
    )
    assert np.allclose(pca_sklearn.explained_variance_, pca_torch.eigenvalues)
    assert np.allclose(pca_sklearn.mean_, pca_torch.mean)

    x_sklearn = rng.normal(size=shape)
    x_torch = torch.from_numpy(x_sklearn.copy())

    x_transformed_sklearn = pca_sklearn.transform(x_sklearn)
    x_transformed_torch = pca_torch.transform(x_torch)
    assert np.allclose(x_transformed_sklearn, x_transformed_torch)

    x_inverse_sklearn = pca_sklearn.inverse_transform(x_transformed_sklearn)
    x_inverse_torch = pca_torch.inverse_transform(x_transformed_torch)
    assert np.allclose(x_inverse_sklearn, x_inverse_torch)


@pytest.mark.parametrize(
    "shape",
    [(5, 100, 10), (1, 20, 20), (2, 10, 100), (5, 100, 1), (5, 2, 10)],
)
@pytest.mark.parametrize("n_components_fraction", [0, 0.25, 0.5, None])
def test_pca_3d(
    shape: tuple[int, ...],
    n_components_fraction: float | None,
) -> None:
    rng = np.random.default_rng(seed=0)
    x_sklearn = rng.normal(size=shape)
    x_torch = torch.from_numpy(x_sklearn.copy())

    if n_components_fraction is not None:
        n_components = int(max(np.floor(n_components_fraction * min(shape[-2:])), 1))
    else:
        n_components = min(shape[-2:])

    if shape[-2] <= shape[-1]:
        if n_components != 1:
            n_components -= 1

    pca_torch = PCA(n_components=n_components)
    pca_torch.fit(x_torch)
    x_torch = torch.from_numpy(x_sklearn.copy())
    x_transformed_torch = pca_torch.transform(x_torch)
    x_inverse_torch = pca_torch.inverse_transform(x_transformed_torch)

    for i_batch in range(shape[0]):
        pca_sklearn = PCASklearn(n_components=n_components)
        pca_sklearn.fit(x_sklearn[i_batch, ...])

        rng = np.random.default_rng(seed=0)
        x_sklearn = rng.normal(size=shape)
        x_transformed_sklearn = pca_sklearn.transform(x_sklearn[i_batch, ...])
        x_inverse_sklearn = pca_sklearn.inverse_transform(x_transformed_sklearn)

        assert np.allclose(
            pca_sklearn.components_,
            pca_torch.eigenvectors[i_batch, ...].transpose(-2, -1),
        )
        assert np.allclose(
            pca_sklearn.explained_variance_,
            pca_torch.eigenvalues[i_batch, ...],
        )
        assert np.allclose(pca_sklearn.mean_, pca_torch.mean[i_batch, ...])
        assert np.allclose(x_transformed_sklearn, x_transformed_torch[i_batch, ...])
        assert np.allclose(x_inverse_sklearn, x_inverse_torch[i_batch, ...])
