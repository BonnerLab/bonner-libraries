import torch


def center(x: torch.Tensor, *, dim: int = 0, nan_policy: str = "omit") -> torch.Tensor:
    match nan_policy:
        case "propagate":
            x_mean = x.mean(dim=dim, keepdim=True)
        case "omit":
            x_mean = x.nanmean(dim=dim, keepdim=True)
        case _:
            raise ValueError("x contains NaNs")

    x = x - x_mean
    return x
