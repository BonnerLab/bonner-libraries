from collections.abc import Callable

import torch


def linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.transpose(-2, -1)


def _double_center(k: torch.Tensor) -> torch.Tensor:
    """Return ``H K H`` with ``H = I - 11^T/n``, without materializing ``H``.

    ``H K H = K - rowmean - colmean + grandmean``. Equivalent to the dense form but
    ``O(n^2)`` instead of ``O(n^3)``, and -- critically -- it inherits ``k``'s device and
    dtype for free, where a ``torch.eye(n)`` literal does not.
    """
    return (
        k
        - k.mean(dim=-2, keepdim=True)
        - k.mean(dim=-1, keepdim=True)
        + k.mean(dim=(-2, -1), keepdim=True)
    )


def hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:  # noqa: E741
    """Biased HSIC_0 (Kornblith et al. 2019) between two kernel matrices.

    Uses ``Tr(KH LH) = Tr(HKH L)`` (trace cyclicity plus ``H`` idempotent), so only one
    matrix needs centering and no dense ``n x n`` centering matrix is built. At n=10k that
    avoids a 400 MB allocation and two needless ``n^3`` matmuls performed purely to center.

    Batched inputs ``(..., n, n)`` are supported; the reduction is over the trailing two
    dims.
    """
    n = k.shape[-1]
    # Tr(A B) = sum(A * B^T); B^T rather than B so this stays exact for a non-symmetric l.
    return (_double_center(k) * l.transpose(-2, -1)).sum(dim=(-2, -1)) / ((n - 1) ** 2)


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Callable = linear_kernel,
) -> torch.Tensor:
    k = kernel(x, x)
    l = kernel(y, y)  # noqa: E741
    return hsic(k, l) / torch.sqrt(hsic(k, k) * hsic(l, l))
