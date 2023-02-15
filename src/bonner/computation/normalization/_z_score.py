import torch


def z_score(
    x: torch.Tensor, *, dim: int = 0, unbiased: bool = True, nan_policy: str = "omit"
) -> torch.Tensor:
    match nan_policy:
        case "propagate":
            x_mean = x.mean(dim=dim, keepdim=True)
            x_std = x.std(dim=dim, keepdim=True, unbiased=unbiased)
        case "omit":
            x_mean = x.nanmean(dim=dim, keepdim=True)
            ddof = 1 if unbiased else 0
            x_std = (
                ((x - x_mean) ** 2).nansum(dim=dim, keepdim=True)
                / (x.shape[dim] - ddof)
            ).sqrt()
        case _:
            raise ValueError("x contains NaNs")

    x = (x - x_mean) / x_std
    return x
