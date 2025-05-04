import scipy
import torch


def svd(
    x: torch.Tensor,
    *,
    n_components: int | None = None,
    randomized: bool = False,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the singular value decomposition (SVD) of a matrix.

    This high-level wrapper performs sets opinionated defaults for computing the
    SVD of a matrix (or matrices) `x` wih shape `(..., m, n)`.

    When `randomized` is `True`, we use :py:func:`torch.svd_lowrank` to
    dramatically speed up computation at the slight expense of accuracy. This is
    recommended when `n_components << min(m, n)`. As recommended by the PyTorch
    implementation, the estimated rank is set at `q = min(2 * n_components, m,
    n)`.

    When `randomized` is `False`, a high-precision SVD is computed using the
    `"gesvd" driver
    <https://netlib.org/lapack/explore-html/d1/d7f/group__gesvd.html>`_. If `x`
    is located on the CPU, we use the slow :py:func:`scipy.linalg.svd`
    implementation as no other implementation supports the "gesvd" driver; if
    `x` is on the GPU, we use the fast :py:func:`torch.linalg.svd`
    implementation.

    Warning: the :py:func:`scipy.linalg.svd` implementation does not support
    batches of matrices as input.

    Parameters
    ----------
    x
        The matrix (or matrices) to be decomposed, of shape (..., m, n).
    n_components, optional
        The number of latent components to return. If None, `n_components` is
        set to `min(m, n)`. Defaults to None.
    randomized, optional
        Whether to use a randomized SVD algorithm. Defaults to False.
    niter, optional
        The number of subspace iterations to use when `randomized` is `True`.
        Defaults to 2.

    Returns
    -------
        A tuple (U, S, V), where x = U @ S @ V.transpose(-2, -1) and U (left
        singular vectors), S (singular values), and V (right singular vectors)
        have shapes (..., m, n_components), (..., n_components,), (..., n,
        n_components) respectively.

    """
    n_samples, n_features = x.shape[-2], x.shape[-1]
    if n_components is None:
        n_components = min(n_samples, n_features)

    device = "cpu" if x.get_device() == -1 else "gpu"

    if randomized:
        u, s, v = torch.svd_lowrank(
            x,
            q=min(2 * n_components, n_samples, n_features),
            niter=niter,
        )
        v_h = v.transpose(-2, -1)
        del v
    elif device == "cpu":
        ndim_matrix = 2
        if x.ndim > ndim_matrix:
            error = "batches of matrices are not supported when on CPU"
            raise ValueError(error)
        u, s, v_h = scipy.linalg.svd(x, full_matrices=False, lapack_driver="gesvd")
    else:
        u, s, v_h = torch.linalg.svd(x, full_matrices=False, driver="gesvd")

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
        case _:
            error = "`u` must be 2- or 3-dimensional"
            raise ValueError(error)

    u *= signs.unsqueeze(-2)
    v_h *= signs.unsqueeze(-1)

    return u, v_h
