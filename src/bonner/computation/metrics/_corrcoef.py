import gc
import more_itertools

import torch

from bonner.computation.normalization import z_score, center


def _helper(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_value: str,
    return_diagonal: bool = True,
    unbiased: bool = True,
    batch_size: int = 1000,
) -> torch.Tensor:
    if x.ndim not in {1, 2, 3}:
        raise ValueError(f"x must have 1, 2 or 3 dimensions (n_dim = {x.ndim})")
    x = x.unsqueeze(1) if x.ndim == 1 else x

    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples_x = x.shape[dim_sample_x]
    n_features_x = x.shape[dim_feature_x]

    if y is not None:
        if y.ndim not in {1, 2, 3}:
            raise ValueError(f"y must have 1, 2 or 3 dimensions (n_dim = {y.ndim})")
        y = y.unsqueeze(1) if y.ndim == 1 else y

        dim_sample_y, dim_feature_y = y.ndim - 2, y.ndim - 1
        n_samples_y = y.shape[dim_sample_y]

        if n_samples_x != n_samples_y:
            raise ValueError(
                f"x and y must have same n_samples (x={n_samples_x}, y={n_samples_y}"
            )

        if return_diagonal:
            n_features_y = y.shape[dim_feature_y]
            if n_features_x != n_features_y:
                raise ValueError(
                    "x and y must have same n_features to return diagonal"
                    f" (x={n_features_x}, y={n_features_y})"
                )
    else:
        y = x
        dim_sample_y = dim_sample_x

    match return_value:
        case "pearson_r":
            x = z_score(x, dim=dim_sample_x, unbiased=unbiased, nan_policy="propagate")
            y = z_score(y, dim=dim_sample_y, unbiased=unbiased, nan_policy="propagate")
        case "covariance":
            x = center(x, dim=dim_sample_x, nan_policy="propagate")
            y = center(y, dim=dim_sample_y, nan_policy="propagate")
        case _:
            raise ValueError()

    try:
        x = (x.transpose(-2, -1) @ y) / (n_samples_x - 1)
        if return_diagonal:
            x = torch.diagonal(x, dim1=-2, dim2=-1)
    except:
        if return_diagonal:
            xs = []
            for batch in more_itertools.chunked(range(n_features_x), n=batch_size):
                x_ = x[..., batch].transpose(-2, -1) @ y[..., batch]
                x_ /= n_samples_x - 1
                xs.append(torch.diagonal(x_, dim1=-2, dim2=-1))
            x = torch.concat(xs, dim=-1)
        else:
            raise ValueError("Tensor is too big to fit in memory")
    x = x.squeeze()

    del y
    gc.collect()
    torch.cuda.empty_cache()

    return x


def pearson_r(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_diagonal: bool = True,
    unbiased: bool = True,
) -> torch.Tensor:
    """Computes Pearson correlation coefficients.

    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise correlations are broadcasted along the batch dimension). If x and y are both specified, pairwise correlations between the columns of x and those of y are computed.

    Args:
        x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
        y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
        return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise correlation matrix, defaults to True

    Returns:
        Pearson correlation coefficients (*, n_features_x, n_features_y)
    """
    return _helper(
        x=x,
        y=y,
        return_value="pearson_r",
        return_diagonal=return_diagonal,
        unbiased=unbiased,
    )


def covariance(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_diagonal: bool = True,
    unbiased: bool = True,
) -> torch.Tensor:
    """Computes covariance.

    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise covariances are broadcasted along the batch dimension). If x and y are both specified, pairwise covariances between the columns of x and those of y are computed.

    Args:
        x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
        y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
        return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise covariance matrix, defaults to True

    Returns:
        covariance matrix (*, n_features_x, n_features_y)
    """
    return _helper(
        x=x,
        y=y,
        return_value="covariance",
        return_diagonal=return_diagonal,
        unbiased=unbiased,
    )
