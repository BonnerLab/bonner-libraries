import itertools
from collections.abc import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from bonner.computation.metrics import pearson_r, spearman_r


def compute_rsm(x: torch.Tensor) -> torch.Tensor:
    return pearson_r(
        x.transpose(-2, -1),
        correction=0,
        return_diagonal=False,
    )


def extract_upper_triangle(rsm: torch.Tensor) -> torch.Tensor:
    x_indices, y_indices = torch.triu_indices(rsm.shape[-1], rsm.shape[-1], offset=1)
    return rsm[..., x_indices, y_indices]


def _get_correlation_function(correlation: str) -> Callable:
    match correlation:
        case "Pearson":
            func = pearson_r
        case "Spearman":
            func = spearman_r
        case _:
            raise ValueError
    return func


def compute_rsa_correlation(
    rsm_x: torch.Tensor,
    rsm_y: torch.Tensor,
    /,
    *,
    correlation: str,
    n_bootstraps: int = 5_000,
    subsample_fraction: float = 0.9,
    seed: int = 0,
    batch_size: int = 500,
) -> tuple[float, torch.Tensor]:
    func = _get_correlation_function(correlation)

    n_stimuli = rsm_x.shape[-1]

    rng = np.random.default_rng(seed=seed)

    r_bootstrapped = []

    for bootstrap_indices in tqdm(
        itertools.batched(range(n_bootstraps), n=batch_size),
        desc="bootstrap",
        leave=False,
    ):
        rsms_x, rsms_y = [], []
        for _ in bootstrap_indices:
            samples = rng.permutation(n_stimuli)[: int(subsample_fraction * n_stimuli)]
            rsms_x.append(rsm_x[samples, :][:, samples])
            rsms_y.append(rsm_y[samples, :][:, samples])

        r_bootstrapped.append(
            func(
                extract_upper_triangle(torch.stack(rsms_x)).T,
                extract_upper_triangle(torch.stack(rsms_y)).T,
                return_diagonal=True,
            ),
        )

    r = float(
        func(
            extract_upper_triangle(rsm_x),
            extract_upper_triangle(rsm_y),
        ),
    )
    return r, torch.concatenate(r_bootstrapped)


def clean_rsm(rsm: torch.Tensor, /, *, indices: np.ndarray) -> np.ndarray:
    rsm = rsm[indices, :][:, indices]
    rsm = torch.triu(rsm, diagonal=1)
    rsm[rsm == 0] = torch.nan
    return rsm.cpu().numpy()
