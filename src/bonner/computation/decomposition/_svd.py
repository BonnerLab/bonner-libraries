import numpy as np
import torch


def svd(
    x: torch.Tensor,
    *,
    truncated: bool,
    n_components: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # if x is on the CPU, use numpy.linalg.svd
    if x.get_device() == -1:
        u, s, v_h = np.linalg.svd(x.cpu().numpy(), full_matrices=False)
        u = torch.from_numpy(u)
        s = torch.from_numpy(s)
        v_h = torch.from_numpy(v_h)
    # if x is on the GPU and truncated is True
    elif truncated:
        torch.manual_seed(seed)
        u, s, v = torch.pca_lowrank(x, center=False, q=n_components)
        v_h = v.transpose(-2, -1)
        del v
        torch.cuda.empty_cache()
    # if x is on the GPU and truncated is False
    else:
        u, s, v_h = torch.linalg.svd(x, full_matrices=False, driver="gesvd")

    u, v_h = svd_flip(u=u, v_h=v_h)
    return u, s, v_h


def svd_flip(
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

    u *= signs.unsqueeze(-2)
    v_h *= signs.unsqueeze(-1)

    return u, v_h
