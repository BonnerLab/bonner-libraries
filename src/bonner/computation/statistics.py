import numpy as np
import numpy.typing as npt


def derange(
    n: int,
    *,
    n_derangements: int,
    batch_size: int = 1000,
    seed: int = 0,
) -> np.ndarray[int]:
    rng = np.random.default_rng(seed=seed)

    batch_size = min(batch_size, n_derangements)

    derangements = np.zeros((0, n), dtype=np.uint64)
    while len(derangements) < n_derangements:
        derangements_ = np.stack(
            [rng.permutation(n).astype(np.uint64) for _ in range(batch_size)],
        )
        derangements_ = derangements_[
            ~np.any(derangements_ == np.arange(n, dtype=np.uint64), axis=1),
            ...,
        ]
        derangements = np.concatenate([derangements, derangements_], axis=0)
        derangements = np.unique(derangements, axis=0)

    return derangements[:n_derangements, ...]


def permutation_test(
    samples: np.ndarray,
    null_distribution: np.ndarray,
    tail: str = "both",
) -> npt.NDArray[np.float64]:
    n = len(null_distribution)
    match tail:
        case "left":
            p_value = ((samples >= null_distribution).sum() + 1) / (n + 1)
        case "right":
            p_value = ((samples <= null_distribution).sum() + 1) / (n + 1)
        case "both":
            samples = np.abs(samples)
            null_distribution = np.abs(null_distribution)
            p_value = ((samples <= null_distribution).sum() + 1) / (n + 1)
        case _:
            error = "tail must be `left`, `right` or `both`"
            raise ValueError(error)
    return p_value
