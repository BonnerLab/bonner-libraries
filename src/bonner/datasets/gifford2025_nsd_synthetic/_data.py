from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from bonner.datasets._utilities import nii
from bonner.datasets.allen2021_natural_scenes import load_brain_mask
from bonner.files import s3

from ._stimuli import load_stimulus_information
from ._utilities import BUCKET_NAME, CACHE_PATH


def load_presentations() -> pd.DataFrame:
    filepath = (
        Path("nsddata") / "experiments" / "nsdsynthetic" / "nsdsynthetic_expdesign.mat"
    )
    s3.download(
        filepath,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH / filepath,
    )
    experiment_design = loadmat(CACHE_PATH / filepath)
    stimulus_id = np.squeeze(experiment_design["masterordering"] - 1)
    stimulus_pattern = np.squeeze(experiment_design["stimpattern"]).astype(bool)
    stimuli = xr.DataArray(
        name="stimulus_index",
        data=stimulus_id,
        dims=("presentation",),
        coords={
            "run": ("presentation", np.empty_like(stimulus_id)),
            "trial": ("presentation", np.empty_like(stimulus_id)),
        },
    )
    cumulative_trials = 0
    for idx, trials in enumerate(stimulus_pattern):
        trials_ = np.where(trials)[0]
        stimuli["run"][cumulative_trials : cumulative_trials + len(trials_)] = idx
        stimuli["trial"][cumulative_trials : cumulative_trials + len(trials_)] = trials_
        cumulative_trials += len(trials_)

    reps: dict[str, int] = {}
    repetitions = np.empty(
        shape=(len(stimuli),),
        dtype=np.uint8,
    )
    for i_stimulus, stimulus in enumerate(stimuli.to_numpy()):
        if stimulus in reps:
            reps[stimulus] += 1
        else:
            reps[stimulus] = 0
        repetitions[i_stimulus] = reps[stimulus]

    stimuli = stimuli.assign_coords(repetition=("presentation", repetitions))
    stimulus_information = load_stimulus_information()

    return (
        stimuli.to_dataframe()
        .join(stimulus_information, on="stimulus_index")
        .drop(columns=["stimulus_index"])
    )


def load_betas(
    *,
    subject: int,
    resolution: Literal["1mm", "1pt8mm"],
    preprocessing: Literal["fithrf", "fithrf_GLMdenoise_RR"],
    z_score: bool,
    neuroid_filter: Sequence[bool] | bool = True,
) -> xr.DataArray:
    filepath = (
        Path("nsddata_betas")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / f"nsdsyntheticbetas_{preprocessing}"
        / "betas_nsdsynthetic.hdf5"
    )
    s3.download(
        filepath,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH / filepath,
    )

    brain_mask = load_brain_mask(subject=subject, resolution=resolution)
    validity = load_validity(subject=subject, resolution=resolution).stack(
        {"neuroid": ("x", "y", "z")},
        create_index=True,
    )

    neuroid_filter = np.logical_and(neuroid_filter, validity)
    neuroid_filter = np.logical_and(
        neuroid_filter,
        brain_mask.stack({"neuroid": ("x", "y", "z")}, create_index=True),
    )
    stimuli = load_presentations()

    betas = (
        xr.load_dataarray(CACHE_PATH / filepath)
        .rename(
            {
                "phony_dim_0": "presentation",
                "phony_dim_1": "z",
                "phony_dim_2": "y",
                "phony_dim_3": "x",
            },
        )
        .transpose("x", "y", "z", "presentation")
        .astype(dtype=np.int16, order="C")
        .stack({"neuroid": ("x", "y", "z")}, create_index=False)
        .sel(neuroid=neuroid_filter)
        .transpose("presentation", "neuroid")
        .astype(dtype=np.float32, order="C")
        .assign_coords(
            coords={
                coord: ("neuroid", neuroid_filter[neuroid_filter][coord].data)
                for coord in ("x", "y", "z")
            }
            | {
                column: ("presentation", stimuli[column].to_numpy())
                for column in stimuli.columns
            },
        )
        .assign_attrs(
            {
                "resolution": resolution,
                "preprocessing": preprocessing,
                "z_score": str(z_score),
                "subject": subject,
            },
        )
    )
    if z_score:
        betas = (betas - betas.mean("presentation")) / betas.std("presentation")
    else:
        betas /= 300
    return betas


def load_validity(
    *,
    subject: int,
    resolution: Literal["1mm", "1pt8mm"],
) -> xr.DataArray:
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / "valid_nsdsynthetic.nii.gz"
    )
    s3.download(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return nii.to_dataarray(CACHE_PATH / filepath, flatten=None).astype(
        dtype=bool,
        order="C",
    )
