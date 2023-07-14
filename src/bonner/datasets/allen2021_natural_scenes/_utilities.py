from collections.abc import Collection, Mapping

import numpy as np
import xarray as xr
from bonner.computation.xarray import groupby_reset

from bonner.datasets._utilities import BONNER_DATASETS_HOME

IDENTIFIER = "allen2021.natural_scenes"
BUCKET_NAME = "natural-scenes-dataset"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def compute_shared_stimulus_ids(
    assemblies: Collection[xr.DataArray], *, n_repetitions: int = 1
) -> set[str]:
    """Gets the IDs of the stimuli shared across all the provided assemblies.

    Args:
        assemblies: assemblies for different subjects
        n_repetitions: minimum number of repetitions for the shared stimuli in each subject

    Returns:
        shared stimulus ids
    """
    try:
        return set.intersection(
            *[
                set(
                    assembly["stimulus_id"].values[
                        (assembly["rep_id"] == n_repetitions - 1).values
                    ]
                )
                for assembly in assemblies
            ]
        )
    except Exception:
        return set.intersection(
            *[set(assembly["stimulus_id"].values) for assembly in assemblies]
        )


def compute_noise_ceiling(
    stimulus_id: xr.DataArray, *, ncsnr: xr.DataArray
) -> xr.DataArray:
    """Compute the noise ceiling for a subject's fMRI data using the method described
    in the NSD Data Manual under the "Conversion of ncsnr to noise ceiling percentages"
    section.

    Args:
        assembly: a subject's neural data

    Returns:
        noise ceilings for all voxels
    """
    groupby = stimulus_id.groupby("stimulus_id")

    counts = np.array([len(reps) for reps in groupby.groups.values()])

    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])

    ncsnr_squared = ncsnr**2
    return (ncsnr_squared / (ncsnr_squared + fraction)).rename("noise ceiling")


def average_betas_across_reps(betas: xr.DataArray) -> xr.DataArray:
    return groupby_reset(
        betas.load()
        .groupby("stimulus_id")
        .mean()
        .assign_attrs(betas.attrs)
        .rename(betas.name),
        groupby_coord="stimulus_id",
        groupby_dim="presentation",
    ).transpose("presentation", "neuroid")


def create_roi_selector(
    *,
    rois: xr.DataArray,
    selectors: Collection[Mapping[str, str]],
) -> np.ndarray:
    selections = []
    for selector in selectors:
        selection = rois.sel(selector).values
        if selection.ndim == 1:
            selection = np.expand_dims(selection, axis=0)
        selections.append(selection)
    selection = np.any(np.concatenate(selections, axis=0), axis=0)
    return selection


def filter_betas_by_stimulus_id(
    betas: xr.DataArray, *, stimulus_ids: set[str], exclude: bool = False
) -> xr.DataArray:
    selection = np.isin(betas["stimulus_id"].values, list(stimulus_ids))
    if exclude:
        selection = ~selection
    return betas.isel({"presentation": selection})
