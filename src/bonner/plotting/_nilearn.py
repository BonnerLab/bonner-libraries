import numpy as np
import numpy.typing as npt


def normalize_curv_map(
    curv_map: npt.NDArray[np.floating],
    /,
    *,
    low: float = 0.25,
    high: float = 0.5,
) -> np.ndarray:
    curv_map = curv_map.copy()
    negative = curv_map < 0
    positive = curv_map >= 0
    curv_map[negative] = low
    curv_map[positive] = high
    return curv_map
