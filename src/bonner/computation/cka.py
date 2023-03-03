from collections.abc import Callable

import torch


def linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.transpose(-2, -1)


def hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    n = k.shape[0]
    h = torch.eye(n) - torch.ones((n, n)) / n

    kh = torch.linalg.matmul(k, h)
    lh = torch.linalg.matmul(l, h)
    return torch.trace(kh @ lh) / ((n - 1) ** 2)


def cka(
    x: torch.Tensor, y: torch.Tensor, kernel: Callable = linear_kernel
) -> torch.Tensor:
    k = kernel(x, x)
    l = kernel(y, y)
    return hsic(k, l) / torch.sqrt(hsic(k, k) * hsic(l, l))
