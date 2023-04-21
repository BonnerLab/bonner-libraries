import torch


def z_score(
    x: torch.Tensor, *, dim: int = 0, correction: float = 1, nan_policy: str = "omit"
) -> torch.Tensor:
    match nan_policy:
        case "propagate":
            x_mean = x.mean(dim=dim, keepdim=True)
            x_std = x.std(dim=dim, keepdim=True, correction=correction)
        case "omit":
            x_mean = x.nanmean(dim=dim, keepdim=True)
            x_std = (
                ((x - x_mean) ** 2).nansum(dim=dim, keepdim=True)
                / (x.shape[dim] - correction)
            ).sqrt()
        case _:
            raise ValueError("x contains NaNs")

    x = (x - x_mean) / x_std
    return x
