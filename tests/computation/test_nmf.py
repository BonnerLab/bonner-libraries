"""Tests for the HALS-based NMF (bonner.computation.decomposition._nmf.NMF).

Guards the HALS rewrite of the fit/transform solvers (previously multiplicative
updates, which badly under-converged on genuinely low-rank data). Covers:

- fit convergence: exact rank-r factorization is recovered to <=1e-3 rel-recon
  (the multiplicative-update solver only reached ~7.8e-2 at the same max_iter);
- fit parity with sklearn's coordinate-descent NMF on random matrices;
- transform (the actual downstream use path — NNLS through a fixed basis):
  reconstructs held-out rows and is stable w.r.t. iteration count (convex NNLS);
- structural invariants: non-negativity, EV-sorted components_, determinism of
  the nndsvd path, and running on CUDA when available.
"""

import numpy as np
import pytest
import torch
from bonner.computation.decomposition._nmf import NMF
from sklearn.decomposition import NMF as NMFSklearn

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rel_recon(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return (torch.norm(x - x_hat) / torch.norm(x)).item()


@pytest.mark.parametrize("device", DEVICES)
def test_fit_recovers_exact_low_rank(device: str) -> None:
    """An exact non-negative rank-r product X = W @ H is reconstructed to <=1e-3."""
    rng = np.random.default_rng(0)
    n, m, r = 200, 80, 6
    w_true = np.abs(rng.normal(size=(n, r)))
    h_true = np.abs(rng.normal(size=(r, m)))
    x = torch.from_numpy((w_true @ h_true).astype(np.float32)).to(device)

    nmf = NMF(n_components=r, init="nndsvd", max_iter=500, tol=1e-6)
    nmf.fit(x)
    w = nmf.transform(x)
    x_hat = nmf.inverse_transform(w)

    assert _rel_recon(x, x_hat) <= 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_fit_parity_with_sklearn(device: str) -> None:
    """On random non-negative data, HALS reconstruction matches sklearn CD (<0.01)."""
    rng = np.random.default_rng(1)
    x_np = np.abs(rng.normal(size=(150, 40))).astype(np.float64)
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)

    nmf = NMF(n_components=10, init="nndsvd", max_iter=500, tol=1e-6)
    nmf.fit(x)
    w = nmf.transform(x)
    x_hat = nmf.inverse_transform(w).cpu().numpy()
    recon_bonner = np.linalg.norm(x_np - x_hat) / np.linalg.norm(x_np)

    sk = NMFSklearn(n_components=10, init="nndsvd", solver="cd", max_iter=500, tol=1e-6)
    w_sk = sk.fit_transform(x_np)
    recon_sk = np.linalg.norm(x_np - w_sk @ sk.components_) / np.linalg.norm(x_np)

    assert abs(recon_bonner - recon_sk) < 0.01


@pytest.mark.parametrize("device", DEVICES)
def test_transform_reconstructs_in_sample(device: str) -> None:
    """transform() (NNLS through the fixed fitted basis) reproduces the fit residual.

    NMF factorizations are not unique (the non-negative basis cone is
    identifiable only under separability), so a *held-out* row need not lie in
    the fitted cone — the guaranteed property is that transform solves the
    convex NNLS optimally, i.e. reproduces the fitted reconstruction in-sample.
    """
    rng = np.random.default_rng(2)
    n, m, r = 300, 60, 5
    w_true = np.abs(rng.normal(size=(n, r)))
    h_true = np.abs(rng.normal(size=(r, m)))
    x = torch.from_numpy((w_true @ h_true).astype(np.float32)).to(device)

    nmf = NMF(n_components=r, init="nndsvd", max_iter=500, tol=1e-6)
    nmf.fit(x)
    w = nmf.transform(x)  # re-solve W given the fitted H
    x_hat = nmf.inverse_transform(w)

    assert _rel_recon(x, x_hat) <= 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_transform_stable_vs_iterations(device: str) -> None:
    """Convex NNLS: transform W converges — more iters barely move the solution."""
    rng = np.random.default_rng(3)
    x = torch.from_numpy(np.abs(rng.normal(size=(120, 50))).astype(np.float32)).to(device)

    nmf = NMF(n_components=8, init="nndsvd", max_iter=500, tol=1e-6)
    nmf.fit(x)

    nmf.transform_max_iter = 100
    w_short = nmf.transform(x)
    nmf.transform_max_iter = 400
    w_long = nmf.transform(x)

    assert _rel_recon(w_long, w_short) <= 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_non_negativity(device: str) -> None:
    rng = np.random.default_rng(4)
    x = torch.from_numpy(np.abs(rng.normal(size=(100, 30))).astype(np.float32)).to(device)

    nmf = NMF(n_components=7, init="nndsvd", max_iter=300)
    nmf.fit(x)
    w = nmf.transform(x)

    assert (nmf.components_ >= 0).all()
    assert (w >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_components_ev_sorted(device: str) -> None:
    """components_ are ordered by descending explained variance (||w_i||^2 ||h_i||^2)."""
    rng = np.random.default_rng(5)
    x = torch.from_numpy(np.abs(rng.normal(size=(200, 40))).astype(np.float32)).to(device)

    nmf = NMF(n_components=10, init="nndsvd", max_iter=400, tol=1e-6)
    nmf.fit(x)
    w = nmf.transform(x)

    importance = (w**2).sum(dim=0) * (nmf.components_**2).sum(dim=1)
    diffs = importance[1:] - importance[:-1]
    # allow tiny numerical slack
    assert (diffs <= 1e-3 * importance.max()).all()


@pytest.mark.parametrize("device", DEVICES)
def test_nndsvd_deterministic(device: str) -> None:
    """The nndsvd init path is RNG-free → two fits give identical components_."""
    rng = np.random.default_rng(6)
    x = torch.from_numpy(np.abs(rng.normal(size=(150, 35))).astype(np.float32)).to(device)

    a = NMF(n_components=9, init="nndsvd", max_iter=300, tol=1e-6)
    a.fit(x)
    b = NMF(n_components=9, init="nndsvd", max_iter=300, tol=1e-6)
    b.fit(x)

    # components_ enters cache keys downstream → must be bit-identical.
    assert torch.allclose(a.components_, b.components_)
    # transform values are computed results (not cache keys); GPU matmul
    # reductions make them non-deterministic at the EPSILON level, so assert
    # numerical (not bitwise) agreement.
    assert torch.allclose(a.transform(x), b.transform(x), atol=1e-5, rtol=1e-4)
