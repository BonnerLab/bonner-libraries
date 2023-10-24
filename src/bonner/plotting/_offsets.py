from collections.abc import Sequence

import pandas as pd


def apply_offset(
    x: pd.DataFrame,
    *,
    keys: Sequence[str],
    offset_key: str,
    offset_magnitude: float,
    offset_type: str = "additive",
    ordering: dict[str, Sequence] | None = None,  # TODO implement ordering the keys
) -> pd.DataFrame:
    groups = x.groupby(keys)
    is_even = bool(len(groups) % 2)

    center: int | float
    if is_even:
        center = len(groups) // 2
    else:
        center = len(groups) // 2 - 0.5

    offset_groups = []
    for i_group, (_, group) in enumerate(groups):
        offset = i_group - center
        match offset_type:
            case "additive":
                group[offset_key] += offset_magnitude * offset
            case "multiplicative":
                group[offset_key] *= offset_magnitude**offset
            case _:
                raise ValueError("offset_type not recognized")
        offset_groups.append(group)
    return pd.concat(offset_groups, axis=0)
