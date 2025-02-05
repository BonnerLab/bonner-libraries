from collections.abc import Collection, Mapping

import numpy as np
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME

IDENTIFIER = "allen2021.natural_scenes"
BUCKET_NAME = "natural-scenes-dataset"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 8


def compute_shared_stimuli(
    assemblies: Collection[xr.DataArray],
    *,
    n_repetitions: int = 1,
) -> set[int]:
    """Get the IDs of the stimuli shared across all the provided assemblies.

    Args:
    ----
        assemblies: assemblies for different subjects
        n_repetitions: minimum number of repetitions for the shared stimuli in each subject

    Returns:
    -------
        shared stimulus ids

    """
    try:
        return set.intersection(
            *[
                set(
                    assembly["stimulus"].data[
                        (assembly["repetition"] == n_repetitions - 1).data
                    ],
                )
                for assembly in assemblies
            ],
        )
    except Exception:
        return set.intersection(
            *[set(assembly["stimulus"].values) for assembly in assemblies],
        )


def compute_noise_ceiling(
    stimuli: xr.DataArray,
    *,
    ncsnr: xr.DataArray,
) -> xr.DataArray:
    """Compute the noise ceiling for a subject's fMRI data using the method described in the NSD Data Manual under the "Conversion of ncsnr to noise ceiling percentages" section.

    Args:
    ----
        stimuli: TODO
        ncsnr: TODO

    Returns:
    -------
        noise ceilings for all voxels

    """
    groupby = stimuli.groupby("stimulus")

    counts = np.array([len(reps) for reps in groupby.groups.values()])

    if counts is None:
        fraction = 1
    else:
        reps = {1: 0, 2: 0, 3: 0}
        unique, counts = np.unique(counts, return_counts=True)
        reps = reps | dict(zip(unique, counts, strict=True))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])

    ncsnr_squared = ncsnr**2
    return (ncsnr_squared / (ncsnr_squared + fraction)).rename("noise ceiling")


def create_roi_selector(
    *,
    rois: xr.DataArray,
    selectors: Collection[Mapping[str, str]],
) -> np.ndarray:
    selections = []
    for selector in selectors:
        selection = rois.sel(selector).data
        if selection.ndim == 1:
            selection = np.expand_dims(selection, axis=0)
        selections.append(selection)
    return np.any(np.concatenate(selections, axis=0), axis=0)
