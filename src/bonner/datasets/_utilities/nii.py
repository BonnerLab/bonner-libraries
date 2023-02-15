from collections.abc import Sequence
from pathlib import Path
import copy

import numpy as np
import xarray as xr
import nibabel as nib


def to_dataarray(
    filepath: Path,
    *,
    dims: Sequence[str] = ("x", "y", "z"),
    flatten: dict[str, Sequence[str]] = {"neuroid": ("x", "y", "z")},
) -> xr.DataArray:
    """Format an NII file as a DataArray.

    Args:
        filepath: path to NII file [must be a 3D array (x, y, z) or 4D array e.g. (presentation, x, y, z)]
        flatten: whether to flatten all the spatial dimensions into a "neuroid" dimension

    Returns:
        brain volume
    """
    nii = nib.load(filepath).get_fdata()

    nii = xr.DataArray(
        data=nii,
        dims=dims,
    )
    nii = nii.assign_coords(
        {dim: (dim, np.arange(nii.sizes[dim], dtype=np.uint8)) for dim in dims}
    )
    if flatten:
        assert len(flatten) == 1
        dim, dims_to_flatten = copy.deepcopy(flatten).popitem()
        nii = nii.stack({dim: dims_to_flatten}, create_index=False)
    return nii
