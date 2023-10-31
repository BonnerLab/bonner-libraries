import numpy as np
from bonner.caching import cache


@cache(identifier="{x} / test.pkl")
def add_1(x: int, y: str = "cat") -> np.ndarray:
    return np.zeros(x)
