import numpy as np
import numpy.typing as npt


def permutation_test(
    samples: np.ndarray, null_distribution: np.ndarray, tail: str = "both"
) -> npt.NDArray[np.float_]:
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
            raise ValueError("tail must be `left`, `right` or `both`")
    return p_value
