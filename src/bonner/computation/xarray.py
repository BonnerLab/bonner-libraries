from typing import Any
from collections.abc import Collection, Hashable

import numpy as np
import numpy.typing as npt
import xarray as xr


def align_source_to_target(
    *,
    source: xr.DataArray,
    target: xr.DataArray,
    sample_coord: str,
    sample_dim: str,
) -> xr.DataArray:
    def _helper(  # type: ignore  # source and target can have Any dtype
        source: npt.NDArray[Any], target: npt.NDArray[Any]
    ) -> npt.NDArray[np.int_]:
        assert len(set(source)) == len(source), "source has duplicate values"
        assert set(target).issubset(
            set(source)
        ), "not all the target elements are present in the source"

        indices = {value: idx for idx, value in enumerate(source)}
        return np.array([indices[target_sample] for target_sample in target])

    indices = _helper(
        source=source[sample_coord].data, target=target[sample_coord].data
    )
    return source.load().isel({sample_dim: indices})


def filter_dataarray(
    array: xr.DataArray, *, coord: str, values: Collection[Any], exclude: bool = False
) -> xr.DataArray:
    filter_ = np.isin(array[coord].data, values)
    if exclude:
        filter_ = ~filter_

    dim = array[coord].dims[0]
    return array.load().isel({dim: filter_})


def groupby_reset(
    x: xr.DataArray, *, groupby_coord: str, groupby_dim: Hashable
) -> xr.DataArray:
    return (
        x.reset_index(groupby_coord)
        .rename({groupby_coord: groupby_dim})
        .assign_coords({groupby_coord: (groupby_dim, x[groupby_coord].values)})
        .drop_vars(groupby_dim)
    )
