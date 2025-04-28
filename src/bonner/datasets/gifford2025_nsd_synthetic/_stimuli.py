from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr
from PIL import Image
from torch.utils.data import Dataset

from bonner.files import s3

from ._utilities import BUCKET_NAME, CACHE_PATH, IDENTIFIER

N_STIMULI = 284
N_STIMULI_SHARED = 220


def load_stimulus_information() -> pd.DataFrame:
    filepath = (
        Path("nsddata")
        / "experiments"
        / "nsdsynthetic"
        / "nsdsyntheticimageinformation.csv"
    )
    s3.download(
        filepath,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH / filepath,
    )
    return (
        pd.read_csv(CACHE_PATH / filepath)
        .drop(columns=["Image number", "Image subclass number", "Image class number"])
        .rename(
            columns={
                "Image": "stimulus",
                "Image subclass": "subclass",
                "Image class": "class",
            },
        )
    )


def load_shared_stimuli() -> xr.DataArray:
    stimulus_information = load_stimulus_information()
    filepath = (
        Path("nsddata_stimuli")
        / "stimuli"
        / "nsdsynthetic"
        / "nsdsynthetic_stimuli.hdf5"
    )
    s3.download(
        filepath,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH / filepath,
    )
    return (
        xr.open_dataarray(CACHE_PATH / filepath)
        .rename(
            {
                "phony_dim_0": "stimulus",
                "phony_dim_1": "height",
                "phony_dim_2": "width",
                "phony_dim_3": "channel",
            },
        )
        .assign_coords(
            {
                "channel": ("channel", ["R", "G", "B"]),
            }
            | {
                coord: (
                    "stimulus",
                    stimulus_information[coord].to_numpy()[:N_STIMULI_SHARED],
                )
                for coord in stimulus_information.columns
            },
        )
        .transpose("stimulus", "channel", "height", "width")
        .rename("stimuli")
    )


def load_unshared_stimuli(subject: int) -> xr.DataArray:
    stimulus_information = load_stimulus_information()
    filepath = (
        Path("nsddata_stimuli")
        / "stimuli"
        / "nsdsynthetic"
        / f"nsdsynthetic_colorstimuli_subj{1 + subject:02}.hdf5"
    )
    s3.download(
        filepath,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH / filepath,
    )
    return (
        xr.open_dataarray(CACHE_PATH / filepath)
        .rename(
            {
                "phony_dim_0": "stimulus",
                "phony_dim_1": "height",
                "phony_dim_2": "width",
                "phony_dim_3": "channel",
            },
        )
        .assign_coords(
            {
                "channel": ("channel", ["R", "G", "B"]),
            }
            | {
                coord: (
                    "stimulus",
                    stimulus_information[coord].to_numpy()[N_STIMULI_SHARED:],
                )
                for coord in stimulus_information.columns
            },
        )
        .transpose("stimulus", "channel", "height", "width")
        .rename("stimuli")
    )


class StimulusSet(Dataset):
    def __init__(self: Self, *, subject: int) -> None:
        self.identifier = IDENTIFIER
        self.stimuli_shared = load_shared_stimuli()
        self.stimuli_unshared = load_unshared_stimuli(subject=subject)

    def __getitem__(
        self: Self,
        stimulus: int | str,
    ) -> Image.Image:
        if isinstance(stimulus, int):
            if stimulus < N_STIMULI_SHARED:
                image = self.stimuli_shared.isel(stimulus=stimulus)
            else:
                image = self.stimuli_unshared.isel(
                    stimulus=stimulus - N_STIMULI_SHARED,
                )
        elif isinstance(stimulus, str):
            try:
                image = self.stimuli_shared.sel(stimulus=stimulus)
            except:
                image = self.stimuli_unshared.sel(
                    stimulus=stimulus,
                )
        else:
            error = "`stimulus` must be int or str"
            raise TypeError(error)

        return Image.fromarray(
            image.transpose("height", "width", "channel").to_numpy(),
        )

    def __len__(self: Self) -> int:
        return N_STIMULI
