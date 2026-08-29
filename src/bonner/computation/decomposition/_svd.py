import torch


def svd(
    x: torch.Tensor,
    *,
    n_components: int,
    randomized: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a truncated SVD with a deterministic sign convention.

    Runs on CPU and CUDA alike. ``torch.linalg.svd`` is deliberately called without ``driver=``:
    that argument is accepted only for CUDA inputs on the cuSOLVER backend and raises on a CPU
    input, so passing it would break every CPU caller.

    ``x`` is never centred, by either solver, so a caller wanting a PCA must centre it first.

    Signs are fixed so that the largest-magnitude entry of each left singular vector is positive,
    which makes the decomposition reproducible across runs and devices. ``scikit-learn``'s ``PCA``
    takes its signs from the right singular vectors instead, so the two agree only up to a
    per-component sign.

    Args:
    ----
        x: matrix to decompose (*, n_samples, n_features), with at most one batch dimension
        n_components: number of leading components to return
        randomized: use the approximate ``torch.pca_lowrank`` solver rather than the exact one

    Returns:
    -------
        left singular vectors (*, n_samples, n_components), singular values (*, n_components),
        and right singular vectors (*, n_features, n_components) — V itself, not its transpose

    """
    if randomized:
        u, s, v = torch.pca_lowrank(x, q=n_components, center=False)
        v_h = v.transpose(-2, -1)
        del v
    else:
        u, s, v_h = torch.linalg.svd(x, full_matrices=False)
    u, v_h = _svd_flip(u=u, v_h=v_h)

    return (
        u[..., :n_components],
        s[..., :n_components],
        v_h.transpose(-2, -1)[..., :n_components],
    )


def _svd_flip(
    *,
    u: torch.Tensor,
    v_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip singular vector pairs so each left singular vector's largest entry is positive.

    ``u`` and ``v_h`` are modified in place and returned. Only 2- and 3-dimensional ``u`` are
    handled; a higher-dimensional input falls through the match with ``signs`` unbound.

    Args:
    ----
        u: left singular vectors (*, n_samples, k)
        v_h: transposed right singular vectors (*, k, n_features)

    Returns:
    -------
        the sign-corrected ``u`` and ``v_h``

    """
    max_abs_cols = torch.argmax(torch.abs(u), dim=-2)
    match u.ndim:
        case 3:
            signs = torch.stack(
                [
                    torch.sign(u[i_batch, max_abs_cols[i_batch, :], range(u.shape[-1])])
                    for i_batch in range(u.shape[0])
                ],
                dim=0,
            )
        case 2:
            signs = torch.sign(u[..., max_abs_cols, range(u.shape[-1])])

    u *= signs.unsqueeze(-2)
    v_h *= signs.unsqueeze(-1)

    return u, v_h
