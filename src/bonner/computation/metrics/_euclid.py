import torch


def euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    /,
    *,
    return_diagonal: bool = True,
    copy: bool = True,
) -> torch.Tensor:
    """Compute Euclidean distances between the feature columns of one or two tensors.

    The sum runs over the sample axis, so the distances are between columns and the samples are
    the coordinates — the reverse of the row-wise reading the argument names suggest. For a
    ``(n_samples, n_features)`` input the full output is ``(n_features, n_features)``.

    The result is squeezed, so any axis of length one disappears; a single-feature input comes
    back as a scalar rather than a ``(1, 1)`` matrix.

    Args:
    ----
        x: features (*, n_samples, n_features_x); a 1-dimensional input is read as one feature
        y: features (*, n_samples, n_features_y); defaults to ``x``, giving self-distances.
            ``n_samples`` must match ``x``, and so must the feature count when
            ``return_diagonal`` is set
        return_diagonal: pair the columns of ``x`` and ``y`` elementwise instead of computing
            every cross pair, which avoids the quadratic intermediate
        copy: clone the inputs before reshaping them

    Returns:
    -------
        distances — ``(*, n_features)`` when ``return_diagonal`` is set, otherwise
        ``(*, n_features_x, n_features_y)``, squeezed

    """
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
                f"x and y must have same n_samples (x={n_samples_x}, y={n_samples_y})"
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

    try:
        if return_diagonal:
            distance = torch.sqrt(((x - y) ** 2).sum(dim=dim_sample_x))
        else:
            x_expanded = x.unsqueeze(-1)
            y_expanded = y.unsqueeze(-2)
            distance = torch.sqrt(((x_expanded - y_expanded) ** 2).sum(dim=dim_sample_x))
            
    except MemoryError:
        error = "Tensor is too big to fit in memory"
        raise ValueError(error) from MemoryError
    
    return distance.squeeze()