from collections.abc import Callable

import torch


def linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of inner products between the rows of two feature matrices.

    Args:
    ----
        x: features (*, n_samples_x, n_features)
        y: features (*, n_samples_y, n_features)

    Returns:
    -------
        kernel matrix (*, n_samples_x, n_samples_y)

    """
    return x @ y.transpose(-2, -1)


def _double_center(k: torch.Tensor) -> torch.Tensor:
    """Return ``H K H`` with ``H = I - 11^T/n``, without materializing ``H``.

    ``H K H = K - rowmean - colmean + grandmean``, which is equivalent to the dense form at
    ``O(n^2)`` rather than ``O(n^3)``. Subtracting means also inherits ``k``'s device and dtype,
    where a ``torch.eye(n)`` literal takes the defaults instead and silently forces a copy.

    Args:
    ----
        k: kernel matrix (*, n, n)

    Returns:
    -------
        the double-centred kernel matrix (*, n, n)

    """
    return (
        k
        - k.mean(dim=-2, keepdim=True)
        - k.mean(dim=-1, keepdim=True)
        + k.mean(dim=(-2, -1), keepdim=True)
    )


def hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:  # noqa: E741
    """Biased HSIC_0 (Kornblith et al. 2019) between two kernel matrices.

    Uses ``Tr(KH LH) = Tr(HKH L)``, from trace cyclicity and the idempotence of ``H``, so only
    one matrix is centred and no dense ``n x n`` centring matrix is built — which avoids both an
    allocation quadratic in ``n`` and two ``n^3`` matmuls performed purely to centre.

    Args:
    ----
        k: kernel matrix (*, n, n)
        l: kernel matrix (*, n, n), batched over the same leading dimensions as ``k``

    Returns:
    -------
        HSIC (*), reduced over the trailing two dimensions

    """
    n = k.shape[-1]
    # Tr(A B) = sum(A * B^T); B^T rather than B so this stays exact for a non-symmetric l.
    return (_double_center(k) * l.transpose(-2, -1)).sum(dim=(-2, -1)) / ((n - 1) ** 2)


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Callable = linear_kernel,
) -> torch.Tensor:
    """Compute centred kernel alignment between two sets of features (Kornblith et al. 2019).

    Both feature matrices must describe the same samples in the same order; they may differ in
    the number of features. The result is invariant to orthogonal transforms and isotropic
    scaling of either input, and lies in ``[0, 1]`` for a positive semi-definite kernel.

    Args:
    ----
        x: features (*, n_samples, n_features_x)
        y: features (*, n_samples, n_features_y)
        kernel: maps a pair of feature matrices to a kernel matrix

    Returns:
    -------
        CKA (*), reduced over samples and features

    """
    k = kernel(x, x)
    l = kernel(y, y)  # noqa: E741
    return hsic(k, l) / torch.sqrt(hsic(k, k) * hsic(l, l))
