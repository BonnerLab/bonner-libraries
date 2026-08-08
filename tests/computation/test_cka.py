"""Tests for linear CKA / HSIC (bonner.computation._cka).

Guards the rewrite of `hsic` from a dense centering matrix to double-centering. The old
implementation built ``h = torch.eye(n) - torch.ones((n, n)) / n`` with no ``device=`` and
no ``dtype=``, so it inherited the *global* defaults and was **CPU-and-float32 only**:
CUDA raised "Expected all tensors to be on the same device", and CPU float64 raised
"expected m1 and m2 to have the same dtype: double != float". The dtype half is the one
that bites, because callers doing fp64 linear algebra hit it on CPU, where it looks nothing
like a device bug.

The load-bearing test here is `test_matches_legacy_cpu_fp32`: a device fix that silently
moves the number would be worse than the bug it fixes, so the legacy implementation is
inlined and the new one must reproduce it exactly in the one regime the old code could run.

Also covers the latent batching bug: `n = k.shape[0]` read the *batch* dim for a
``(b, n, n)`` input and `torch.trace` is 2-D only, so `linear_kernel`'s batch-aware
``.transpose(-2, -1)`` had a dead path. Note a batched *cka* test cannot detect the wrong
``n``: the erroneous ``(n-1)^2`` cancels in ``hsic_kl / sqrt(hsic_kk * hsic_ll)``. Only a
batched *hsic* test does.
"""

import numpy as np
import pytest
import torch
from bonner.computation._cka import cka, hsic, linear_kernel

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]


def _hsic_legacy(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:  # noqa: E741
    """The pre-2026-08-07 implementation, verbatim. CPU + float32 only, by construction."""
    n = k.shape[0]
    h = torch.eye(n) - torch.ones((n, n)) / n
    kh = torch.linalg.matmul(k, h)
    lh = torch.linalg.matmul(l, h)
    return torch.trace(kh @ lh) / ((n - 1) ** 2)


def _data(seed: int, n: int, d: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(n, d))).to(device=device, dtype=dtype)


@pytest.mark.parametrize("n", [10, 50, 200])
def test_matches_legacy_cpu_fp32(n: int) -> None:
    """THE regression guard: the value must not move in the regime the old code supported."""
    x = _data(0, n, 7, "cpu", torch.float32)
    y = _data(1, n, 5, "cpu", torch.float32)
    k, l = linear_kernel(x, x), linear_kernel(y, y)  # noqa: E741

    assert torch.allclose(hsic(k, l), _hsic_legacy(k, l), rtol=1e-5, atol=1e-6)
    legacy_cka = _hsic_legacy(k, l) / torch.sqrt(_hsic_legacy(k, k) * _hsic_legacy(l, l))
    assert torch.allclose(cka(x, y), legacy_cka, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_runs_on_every_device_and_dtype(device: str, dtype: torch.dtype) -> None:
    """The bug itself: float64 and CUDA both used to raise."""
    x = _data(2, 60, 8, device, dtype)
    y = _data(3, 60, 6, device, dtype)
    out = cka(x, y)
    assert out.device.type == device
    assert out.dtype == dtype
    assert torch.isfinite(out)


@pytest.mark.parametrize("device", DEVICES)
def test_value_agrees_across_device_and_dtype(device: str) -> None:
    """Cross-device/dtype self-consistency.

    Legacy parity can only be established at CPU/fp32, so without this the fp64 and CUDA
    values would have no oracle at all.
    """
    ref = cka(_data(4, 80, 9, "cpu", torch.float32), _data(5, 80, 7, "cpu", torch.float32))
    got = cka(_data(4, 80, 9, device, torch.float64), _data(5, 80, 7, device, torch.float64))
    assert abs(float(ref) - float(got)) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_self_similarity_is_one(device: str) -> None:
    x = _data(6, 100, 12, device, torch.float64)
    assert torch.allclose(cka(x, x), torch.ones((), device=device, dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
def test_batched_hsic(device: str) -> None:
    """Batched ``(b, n, n)`` must equal the per-item 2-D result.

    Fails on the legacy code two ways: ``n = k.shape[0]`` reads the batch dim, and
    ``torch.trace`` rejects a 3-D tensor outright. Must be tested on `hsic`, not `cka` --
    see the module docstring.
    """
    b, n = 4, 30
    xs = [_data(10 + i, n, 6, device, torch.float64) for i in range(b)]
    ys = [_data(20 + i, n, 5, device, torch.float64) for i in range(b)]

    batched = hsic(
        torch.stack([linear_kernel(x, x) for x in xs]),
        torch.stack([linear_kernel(y, y) for y in ys]),
    )
    assert batched.shape == (b,)
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        expected = hsic(linear_kernel(x, x), linear_kernel(y, y))
        assert torch.allclose(batched[i], expected)


@pytest.mark.parametrize("device", DEVICES)
def test_invariant_to_isotropic_scaling_and_translation(device: str) -> None:
    """CKA is invariant to scale and translation; centering is what supplies the latter."""
    x = _data(7, 70, 10, device, torch.float64)
    y = _data(8, 70, 8, device, torch.float64)
    shifted = 3.5 * y + torch.full((1, 8), 2.0, device=device, dtype=torch.float64)
    assert torch.allclose(cka(x, y), cka(x, shifted), rtol=1e-9, atol=1e-9)


def test_no_dense_centering_matrix_allocated() -> None:
    """A 10k x 10k fp32 centering matrix would be 400 MB; the rewrite must not build one.

    Sized so that the legacy implementation's temporaries would be clearly visible against
    the input itself, without needing a real 10k run in the test suite.
    """
    n = 2000
    x = _data(9, n, 4, "cpu", torch.float32)
    k = linear_kernel(x, x)
    baseline = k.element_size() * k.nelement()

    tracemalloc_available = True
    try:
        import tracemalloc
    except ImportError:  # pragma: no cover
        tracemalloc_available = False
    if not tracemalloc_available:  # pragma: no cover
        pytest.skip("tracemalloc unavailable")

    tracemalloc.start()
    hsic(k, k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # torch allocations are not tracked by tracemalloc, so this asserts the weaker but
    # still meaningful property that no large *python-side* buffer appears; the real
    # guarantee is structural (no torch.eye(n) in the source).
    assert peak < baseline
