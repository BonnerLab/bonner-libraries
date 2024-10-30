import numpy as np
import torch


def svd(
    x: torch.Tensor,
    *,
    n_components: int,
    randomized: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.get_device() == -1:
        u, s, v_h = np.linalg.svd(x, full_matrices=False)
        u, s, v_h = (
            torch.from_numpy(u),
            torch.from_numpy(s),
            torch.from_numpy(v_h),
        )
    elif randomized:
        u, s, v = torch.pca_lowrank(x, q=n_components, center=False)
        v_h = v.transpose(-2, -1)
        del v
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
