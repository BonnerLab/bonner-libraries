import torch


def _helper(
    x: torch.Tensor,
    y: torch.Tensor | None,
    *,
    center: bool,
    scale: bool,
    correction: float = 1,
    return_diagonal: bool = True,
    copy: bool = True,
) -> torch.Tensor:
    if copy:
        x = torch.clone(x)

    if x.ndim not in {1, 2, 3}:
        raise ValueError(f"x must have 1, 2 or 3 dimensions (n_dim = {x.ndim})")
    x = x.unsqueeze(1) if x.ndim == 1 else x

    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples_x = x.shape[dim_sample_x]
    n_features_x = x.shape[dim_feature_x]

    if y is not None:
        if copy:
            y = torch.clone(y)
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

    if center:
        x -= x.mean(dim=dim_sample_x, keepdim=True)
        y -= y.mean(dim=dim_sample_y, keepdim=True)
    if scale:
        x /= x.std(dim=dim_sample_x, keepdim=True, correction=correction)
        y /= y.std(dim=dim_sample_y, keepdim=True, correction=correction)

    try:
        if return_diagonal:
            x = (x * y).sum(dim=-2) / (n_samples_x - 1)
        else:
            x = (x.transpose(-2, -1) @ y) / (n_samples_x - 1)
    except:
        raise ValueError("Tensor is too big to fit in memory")
    x = x.squeeze()

    return x


def pearson_r(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_diagonal: bool = True,
    correction: bool = 1,
    copy: bool = True,
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
        center=True,
        scale=True,
        correction=correction,
        return_diagonal=return_diagonal,
        copy=copy,
    )


def covariance(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_diagonal: bool = True,
    correction: bool = 1,
    copy: bool = True,
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
        center=True,
        scale=False,
        correction=correction,
        return_diagonal=return_diagonal,
        copy=copy,
    )
