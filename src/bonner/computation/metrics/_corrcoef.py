import torch


def _helper(
    x: torch.Tensor,
    y: torch.Tensor | None,
    /,
    *,
    center: bool,
    scale: bool,
    correction: int = 1,
    return_diagonal: bool = True,
    copy: bool = True,
) -> torch.Tensor:
    if copy:
        x = torch.clone(x)

    if x.ndim not in {1, 2, 3}:
        error = f"x must have 1, 2 or 3 dimensions (n_dim = {x.ndim})"
        raise ValueError(error)
    x = x.unsqueeze(1) if x.ndim == 1 else x

    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples_x = x.shape[dim_sample_x]
    n_features_x = x.shape[dim_feature_x]

    if y is not None:
        if copy:
            y = torch.clone(y)
        if y.ndim not in {1, 2, 3}:
            error = f"y must have 1, 2 or 3 dimensions (n_dim = {y.ndim})"
            raise ValueError(error)
        y = y.unsqueeze(1) if y.ndim == 1 else y

        dim_sample_y, dim_feature_y = y.ndim - 2, y.ndim - 1
        n_samples_y = y.shape[dim_sample_y]

        if n_samples_x != n_samples_y:
            error = (
                f"x and y must have same n_samples (x={n_samples_x}, y={n_samples_y}"
            )
            raise ValueError(error)

        if return_diagonal:
            n_features_y = y.shape[dim_feature_y]
            if n_features_x != n_features_y:
                error = (
                    "x and y must have same n_features to return diagonal"
                    f" (x={n_features_x}, y={n_features_y})"
                )
                raise ValueError(error)
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
        # TODO: batch operation
        if return_diagonal:
            x = (x * y).sum(dim=-2) / (n_samples_x - 1)
        else:
            x = (x.transpose(-2, -1) @ y) / (n_samples_x - 1)
    except MemoryError:
        error = "Tensor is too big to fit in memory"
        raise ValueError(error) from MemoryError
    return x.squeeze()


def pearson_r(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    /,
    *,
    return_diagonal: bool = True,
    correction: int = 1,
    copy: bool = True,
) -> torch.Tensor:
    """Compute Pearson correlation coefficients.

    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise correlations are broadcasted along the batch dimension). If x and y are both specified, pairwise correlations between the columns of x and those of y are computed.

    Args:
    ----
        x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
        y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
        return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise correlation matrix, defaults to True
        correction: Bessel's correction
        copy: whether to copy x and y (otherwise, values may be mutated)

    Returns:
    -------
        Pearson correlation coefficients (*, n_features_x, n_features_y)

    """
    return _helper(
        x,
        y,
        center=True,
        scale=True,
        correction=correction,
        return_diagonal=return_diagonal,
        copy=copy,
    )


def spearman_r(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    return_diagonal: bool = True,
    correction: int = 1,
    copy: bool = True,
) -> torch.Tensor:
    rank_x = x.argsort(dim=0).argsort(dim=0).float()

    if y is not None:
        rank_y = y.argsort(dim=0).argsort(dim=0).float()

    return _helper(
        rank_x,
        rank_y,
        center=True,
        scale=True,
        correction=correction,
        return_diagonal=return_diagonal,
        copy=copy,
    )


def covariance(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    /,
    *,
    return_diagonal: bool = True,
    correction: int = 1,
    copy: bool = True,
) -> torch.Tensor:
    """Compute covariance.

    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise covariances are broadcasted along the batch dimension). If x and y are both specified, pairwise covariances between the columns of x and those of y are computed.

    Args:
    ----
        x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
        y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
        return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise covariance matrix, defaults to True
        correction: Bessel's correction
        copy: whether to copy x and y (otherwise, values may be mutated)

    Returns:
    -------
        covariance matrix (*, n_features_x, n_features_y)

    """
    return _helper(
        x,
        y,
        center=True,
        scale=False,
        correction=correction,
        return_diagonal=return_diagonal,
        copy=copy,
    )
    

def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    /,
    *,
    return_diagonal: bool = True,
    copy: bool = True,
) -> torch.Tensor:
    """Compute cosine similarity.

    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise cosine similarities are broadcasted along the batch dimension). If x and y are both specified, pairwise similarities between the columns of x and those of y are computed.

    Args:
    ----
        x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
        y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
        return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise similarity matrix, defaults to True
        copy: whether to copy x and y (otherwise, values may be mutated)

    Returns:
    -------
        Cosine similarity matrix (*, n_features_x, n_features_y)

    """
    return _helper(
        x,
        y,
        center=False,
        scale=True,
        return_diagonal=return_diagonal,
        copy=copy,
    )
