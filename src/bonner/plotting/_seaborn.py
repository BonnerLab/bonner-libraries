import numpy as np
import pandas as pd


def simulate_errors(
    x: pd.DataFrame,
    *,
    key_value: str = "test covariance",
    key_error: str = "error",
    ddof: int = 0,
) -> pd.DataFrame:
    errors = x[key_error].to_numpy()

    delta = errors * np.sqrt((3 - ddof) / 2)
    return pd.concat(
        [
            x.assign(**{key_value: x[key_value] - delta}),
            x,
            x.assign(**{key_value: x[key_value] + delta}),
        ],
        axis=0,
    )
