import itertools
from pathlib import Path
from typing import Literal

import mat73
import numpy as np
import pandas as pd
import scipy
import xarray as xr

from bonner.datasets.papale2025_tvsd._utilities import (
    CACHE_PATH,
    IDENTIFIER,
)
from bonner.files import download_from_url

ROIS = {
    "F": {
        "V1": list(range(512)),
        "V4": list(range(512, 832)),
        "IT": list(range(832, 1024)),
    },
    "N": {
        "V1": list(range(512)),
        "V4": list(range(512, 768)),
        "IT": list(range(768, 1024)),
    },
}

URL_ROOT = "https://gin.g-node.org/paolo_papale/TVSD/raw/master"


def download(
    *,
    monkey: Literal["F", "N"],
    normalized: bool = False,
    force: bool = False,
) -> Path:
    suffix = "normMUA" if normalized else "MUA_trials"
    url = f"{URL_ROOT}/monkey{monkey}/THINGS_{suffix}.mat"
    return download_from_url(
        url,
        filepath=CACHE_PATH / "download" / f"monkey_{monkey}" / f"THINGS_{suffix}.mat",
        force=force,
    )


def download_electrode_mapping(
    *,
    monkey: Literal["F", "N"],
    force: bool = False,
) -> Path:
    url = f"https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkey{monkey}/_logs/1024chns_mapping_20220105.mat"
    return download_from_url(
        url,
        filepath=CACHE_PATH
        / "download"
        / f"monkey_{monkey}"
        / "mapping_electrodes.mat",
        force=force,
    )


def download_stimulus_mapping(
    *,
    monkey: Literal["F", "N"],
    force: bool = False,
) -> Path:
    url = f"https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkey{monkey}/_logs/things_imgs.mat"
    return download_from_url(
        url,
        filepath=CACHE_PATH / "download" / f"monkey_{monkey}" / "mapping_stimuli.mat",
        force=force,
    )


def _get_rois(*, monkey: Literal["F", "N"]) -> list[str]:
    return list(
        itertools.chain(
            *[[roi] * len(electrodes) for roi, electrodes in ROIS[monkey].items()],
        ),
    )


def _extract_stimulus_ids(paths: list[str]) -> list[str]:
    return [x.split("\\")[-1][:-4] for x in paths]


def load_electrode_metadata(*, monkey: Literal["F", "N"]) -> xr.DataArray:
    data = mat73.loadmat(download(monkey=monkey, normalized=True))
    return xr.Dataset(
        {
            "SNR": xr.DataArray(
                name="SNR",
                data=data["SNR"],
                dims=("neuroid", "day"),
            ),
            "latency": xr.DataArray(
                name="latency",
                data=data["lats"],
                dims=("neuroid", "day"),
            ),
        },
        coords={
            "region": ("neuroid", _get_rois(monkey=monkey)),
        },
    )


def load_normalized_data(
    *,
    monkey: Literal["F", "N"],
    train: bool = True,
) -> xr.DataArray:
    data = mat73.loadmat(download(monkey=monkey, normalized=True))
    stimulus_mapping = mat73.loadmat(download_stimulus_mapping(monkey=monkey))
    regions = _get_rois(monkey=monkey)

    identifier = f"{IDENTIFIER}.monkey={monkey}.normalized=True.train={train}"
    if train:
        return xr.DataArray(
            name=identifier,
            data=data["train_MUA"].transpose(),
            dims=("presentation", "neuroid"),
            coords={
                "stimulus": (
                    "presentation",
                    _extract_stimulus_ids(
                        stimulus_mapping["train_imgs"]["things_path"],
                    ),
                ),
                "region": ("neuroid", regions),
            },
        ).sortby("stimulus")

    return (
        xr.DataArray(
            name=identifier,
            data=data["test_MUA_reps"].transpose(1, 2, 0),
            dims=("stimulus", "repetition", "neuroid"),
            coords={
                "stimulus": (
                    "stimulus",
                    _extract_stimulus_ids(stimulus_mapping["test_imgs"]["things_path"]),
                ),
                "repetition": ("repetition", np.arange(30)),
                "region": ("neuroid", regions),
            },
        )
        .sortby("stimulus")
        .stack(presentation=["stimulus", "repetition"])
        .transpose("presentation", "neuroid")
    )


def load_time_course(*, monkey: Literal["F", "N"]) -> xr.DataArray:
    raise NotImplementedError

    dataset = mat73.loadmat(
        download(monkey=monkey, normalized=False),
        only_include=["ALLMAT", "tb"],
    )

    mapping = (
        np.squeeze(
            scipy.io.loadmat(download_electrode_mapping(monkey=monkey))["mapping"],
        )
        - 1
    )
    rois = _get_rois(monkey=monkey)
    rois = np.array(rois)[np.argsort(mapping)]

    stimuli = (
        pd.DataFrame(
            dataset["ALLMAT"][:, [1, 2, 4, 5]],
            columns=[
                "train_idx",
                "test_idx",
                "count",
                "day",
            ],
            dtype=np.uint32,
        )
        .assign(stimulus=lambda x: x["train_idx"] + x["test_idx"] - 1)
        .assign(training_set=lambda x: x["train_idx"] > 0)
        .drop(columns=["train_idx", "test_idx"])
    )
    return xr.DataArray(
        name=f"{IDENTIFIER}.monkey={monkey}.normalized=False",
        data=mat73.loadmat(
            download(monkey=monkey, normalized=False),
            only_include=["ALLMUA"],
        )["ALLMUA"],
        dims=("electrode", "presentation", "time"),
        coords={
            "region": ("electrode", rois),
            "time": ("time", dataset["tb"]),
        }
        | {
            column: ("presentation", stimuli[column].to_numpy())
            for column in stimuli.columns
        },
    )
