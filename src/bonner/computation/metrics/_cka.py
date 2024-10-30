from collections.abc import Callable

import torch


def linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.transpose(-2, -1)


def hsic(k_x: torch.Tensor, k_y: torch.Tensor) -> torch.Tensor:
    n = k_x.shape[0]
    h = torch.eye(n) - torch.ones((n, n)) / n
    return torch.trace((k_x @ h) @ (k_y @ h)) / ((n - 1) ** 2)


def cka(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Callable = linear_kernel,
) -> torch.Tensor:
    k_x, k_y = kernel(x, x), kernel(y, y)
    return hsic(k_x, k_y) / torch.sqrt(hsic(k_x, k_x) * hsic(k_y, k_y))
