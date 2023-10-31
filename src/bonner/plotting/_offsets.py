from collections.abc import Sequence

import pandas as pd


def apply_offset(
    x: pd.DataFrame,
    *,
    keys: Sequence[str],
    offset_key: str,
    offset_magnitude: float,
    offset_type: str = "additive",
) -> pd.DataFrame:
    groups = x.groupby(keys, sort=False)
    is_even = bool(len(groups) % 2)

    center = len(groups) // 2 if is_even else len(groups) // 2 - 0.5

    offset_groups = []
    for i_group, (_, group) in enumerate(groups):
        offset = i_group - center
        match offset_type:
            case "additive":
                group[offset_key] += offset_magnitude * offset
            case "multiplicative":
                group[offset_key] *= offset_magnitude**offset
            case _:
                error = "offset_type not recognized"
                raise ValueError(error)
        offset_groups.append(group)
    return pd.concat(offset_groups, axis=0)
