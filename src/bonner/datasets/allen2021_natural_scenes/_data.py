from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from bonner.datasets._utilities import nii
from bonner.datasets.allen2021_natural_scenes._stimuli import load_nsd_metadata
from bonner.datasets.allen2021_natural_scenes._utilities import (
    BUCKET_NAME,
    CACHE_PATH,
)
from bonner.files import download_from_s3

RESOLUTION = "1pt8mm"
PREPROCESSING = "fithrf_GLMdenoise_RR"
N_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)
N_RUNS_PER_SESSION = 12
N_TRIALS_PER_SESSION = 750
ROI_SOURCES = {
    "surface": (
        "streams",
        "prf-visualrois",
        "prf-eccrois",
        "floc-places",
        "floc-faces",
        "floc-bodies",
        "floc-words",
        "HCP_MMP1",
        "Kastner2015",
        "nsdgeneral",
        "corticalsulc",
    ),
    "volume": ("MTL", "thalamus"),
}


def load_brain_mask(*, subject: int, resolution: str) -> xr.DataArray:
    """Load and format a Boolean brain mask for the functional data.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
    -------
        Boolean brain mask

    """
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / "brainmask.nii.gz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return nii.to_dataarray(CACHE_PATH / filepath, flatten=None).astype(bool, order="C")


def load_validity(*, subject: int, resolution: str) -> xr.DataArray:
    validity = []
    n_sessions = N_SESSIONS[subject]

    sessions = {
        f"nsd-{session}": f"session{session + 1:02}" for session in range(n_sessions)
    } | {"prffloc": "prffloc"}

    for session, suffix in sessions.items():
        filepath = (
            Path("nsddata")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"valid_{suffix}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
        validity.append(
            nii.to_dataarray(CACHE_PATH / filepath, flatten=None)
            .expand_dims({"session": [session]})
            .astype(dtype=bool, order="C"),
        )
    return xr.concat(validity, dim="session")


def load_betas(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
    z_score: bool,
    neuroid_filter: Sequence[bool] | bool = True,
) -> xr.DataArray:
    """Load betas.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"
        preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf"

    Returns:
    -------
        betas

    """

    def load_presentations(subject: int) -> xr.DataArray:
        metadata = load_nsd_metadata()
        metadata = np.array(
            metadata.loc[
                :,
                [f"subject{subject}_" in column for column in metadata.columns],
            ],
        )
        assert metadata.shape[-1] == 3
        indices = np.nonzero(metadata)
        trials = metadata[indices] - 1  # fix 1-indexing

        sessions = trials // N_TRIALS_PER_SESSION
        intra_session_trials = trials % N_TRIALS_PER_SESSION

        stimuli = xr.DataArray(
            data=np.empty((max(N_SESSIONS), N_TRIALS_PER_SESSION), dtype=np.uint32),
            dims=("session", "trial"),
        )
        stimuli.values[sessions, intra_session_trials] = indices[0]

        stimuli = stimuli.assign_coords(
            {
                "session": np.arange(stimuli.sizes["session"], dtype=np.uint8),
                "trial": np.arange(stimuli.sizes["trial"], dtype=np.uint16),
            },
        ).stack({"presentation": ("session", "trial")}, create_index=True)
        stimuli = stimuli.isel(
            presentation=stimuli["session"].data < N_SESSIONS[subject],
        )

        reps: dict[str, int] = {}
        repetitions = np.empty(
            N_SESSIONS[subject] * N_TRIALS_PER_SESSION,
            dtype=np.uint8,
        )
        for i_stimulus, stimulus in enumerate(stimuli.data):
            if stimulus in reps:
                reps[stimulus] += 1
            else:
                reps[stimulus] = 0
            repetitions[i_stimulus] = reps[stimulus]

        runs: list[int] | np.ndarray = []
        for i_run in range(N_RUNS_PER_SESSION):
            n_trials = 63 if i_run % 2 == 0 else 62
            runs.extend([i_run] * n_trials)
        runs = np.tile(np.array(runs).astype(np.uint8), N_SESSIONS[subject])

        stimuli = stimuli.assign_coords(
            {
                "stimulus": ("presentation", stimuli.data),
                "repetition": ("presentation", repetitions),
                "run": ("presentation", runs),
            },
        )
        return stimuli["stimulus"]

    n_sessions = N_SESSIONS[subject]
    stimuli = load_presentations(subject=subject)
    n_trials = len(stimuli)

    brain_mask = load_brain_mask(subject=subject, resolution=resolution)
    validity = (
        load_validity(subject=subject, resolution=resolution)
        .stack({"neuroid": ("x", "y", "z")}, create_index=True)
        .isel({"session": np.arange(n_sessions)})
        .all("session")
    )

    neuroid_filter = np.logical_and(neuroid_filter, validity)
    neuroid_filter = np.logical_and(
        neuroid_filter,
        brain_mask.stack({"neuroid": ("x", "y", "z")}, create_index=True),
    )

    betas: list[xr.DataArray] | xr.DataArray = []

    betas = xr.DataArray(
        name="beta",
        data=np.empty(shape=(n_trials, neuroid_filter.sum().data), dtype=np.float32),
        dims=("presentation", "neuroid"),
        coords={
            coord: ("neuroid", neuroid_filter[neuroid_filter][coord].data)
            for coord in ("x", "y", "z")
        }
        | {
            coord: ("presentation", stimuli[coord].data)
            for coord in stimuli.reset_index("presentation").coords
        },
        attrs={
            "resolution": resolution,
            "preprocessing": preprocessing,
            "z_score": str(z_score),
            "subject": subject,
        },
    )

    for session in tqdm(np.arange(n_sessions), desc="session"):
        filepath = (
            Path("nsddata_betas")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"betas_{preprocessing}"
            / f"betas_session{session + 1:02}.hdf5"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)

        betas_session = (
            xr.load_dataset(CACHE_PATH / filepath)["betas"]
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
        )

        if z_score:
            betas_session = (
                betas_session - betas_session.mean("presentation")
            ) / betas_session.std("presentation")
        else:
            betas_session /= 300

        betas.data[
            session * N_TRIALS_PER_SESSION : (session + 1) * N_TRIALS_PER_SESSION,
            :,
        ] = betas_session

    return betas


def load_ncsnr(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
) -> xr.DataArray:
    """Load and format noise-ceiling signal-to-noise ratios (NCSNR).

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"
        preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf

    Returns:
    -------
        noise-ceiling SNRs

    """
    filepath = (
        Path("nsddata_betas")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / f"betas_{preprocessing}"
        / "ncsnr.nii.gz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return nii.to_dataarray(CACHE_PATH / filepath).astype(dtype=np.float64, order="C")


def load_structural_scans(*, subject: int, resolution: str) -> xr.DataArray:
    """Load and format the structural scans registered to the functional data.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
    -------
        structural scans

    """
    scans = []
    sequences = np.array(("T1", "T2", "SWI", "TOF"))
    for sequence in sequences:
        filepath = (
            Path("nsddata")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"{sequence}_to_func{resolution}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
        scans.append(
            nii.to_dataarray(CACHE_PATH / filepath, flatten=None)
            .expand_dims("sequence", axis=0)
            .astype(dtype=np.uint16, order="C"),
        )
    return xr.concat(scans, dim="sequence").assign_coords(
        {"sequence": ("sequence", sequences)},
    )


def load_rois(*, subject: int, resolution: str) -> xr.DataArray:
    """Load the ROI masks for a subject.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
    -------
        ROI masks

    """
    rois = []
    for space, sources in ROI_SOURCES.items():
        for source in sources:
            match space:
                case "surface":
                    filepath = (
                        Path("nsddata")
                        / "freesurfer"
                        / f"subj{subject + 1:02}"
                        / "label"
                        / f"{source}.mgz.ctab"
                    )
                case "volume":
                    filepath = Path("nsddata") / "templates" / f"{source}.ctab"

            download_from_s3(
                filepath,
                bucket=BUCKET_NAME,
                local_path=CACHE_PATH / filepath,
            )

            mapping = (
                pd.read_csv(
                    CACHE_PATH / filepath,
                    sep=r"\s+",
                    names=("label", "roi"),
                )
                .set_index("roi")
                .to_dict()["label"]
            )

            volumes = {}
            for hemisphere in ("lh", "rh"):
                filepath = (
                    Path("nsddata")
                    / "ppdata"
                    / f"subj{subject + 1:02}"
                    / f"func{resolution}"
                    / "roi"
                    / f"{hemisphere}.{source}.nii.gz"
                )
                download_from_s3(
                    filepath,
                    bucket=BUCKET_NAME,
                    local_path=CACHE_PATH / filepath,
                )
                volumes[hemisphere] = nii.to_dataarray(CACHE_PATH / filepath)

                for roi, label in mapping.items():
                    if label != 0:
                        rois.append(
                            (volumes[hemisphere] == label)
                            .expand_dims(roi=[roi], axis=0)
                            .assign_coords(
                                {
                                    "source": ("roi", [source]),
                                    "space": ("roi", [space]),
                                    "hemisphere": ("roi", [hemisphere[0]]),
                                },
                            )
                            .astype(bool, order="C"),
                        )
    rois = xr.concat(rois, dim="roi")
    rois["label"] = rois["roi"].astype(str)
    return rois.drop_vars("roi").set_index({"roi": ("source", "label", "hemisphere")})


def load_receptive_fields(*, subject: int, resolution: str) -> xr.DataArray:
    """Load population receptive field mapping data.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
    -------
        pRF data

    """
    prf_data = []
    quantities = np.array(
        (
            "angle",
            "eccentricity",
            "exponent",
            "gain",
            "R2",
            "size",
        ),
    )
    for quantity in quantities:
        filepath = (
            Path("nsddata")
            / "ppdata"
            / f"subj{subject + 1:02}"
            / f"func{resolution}"
            / f"prf_{quantity}.nii.gz"
        )
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
        prf_data.append(
            nii.to_dataarray(CACHE_PATH / filepath)
            .expand_dims("quantity", axis=0)
            .astype(dtype=np.float64, order="C"),
        )
    return xr.concat(prf_data, dim="quantity").assign_coords(
        {"quantity": ("quantity", quantities)},
    )


def load_functional_contrasts(*, subject: int, resolution: str) -> xr.DataArray:
    """Load functional contrasts.

    Args:
    ----
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
    -------
        functional contrasts

    """
    categories = {}
    for filename in ("domains", "categories"):
        filepath = Path("nsddata") / "experiments" / "floc" / f"{filename}.tsv"
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)

        categories[filename] = list(
            pd.read_csv(CACHE_PATH / filepath, sep="\t").iloc[:, 0].values,
        )

    categories_combined = categories["domains"] + categories["categories"]
    superordinate = np.array(
        [category in categories["domains"] for category in categories_combined],
        dtype=bool,
    )
    floc_data = {}
    for category in categories_combined:
        floc_data[category] = []
        for metric in ("tval", "anglemetric"):
            filepath = (
                Path("nsddata")
                / "ppdata"
                / f"subj{subject + 1:02}"
                / f"func{resolution}"
                / f"floc_{category}{metric}.nii.gz"
            )
            download_from_s3(
                filepath,
                bucket=BUCKET_NAME,
                local_path=CACHE_PATH / filepath,
            )

            floc_data[category].append(
                nii.to_dataarray(CACHE_PATH / filepath)
                .expand_dims(
                    {
                        "category": [category],
                        "metric": [metric],
                    },
                    axis=(0, 1),
                )
                .astype(np.float64, order="C"),
            )
        floc_data[category] = xr.concat(floc_data[category], dim="metric")
    floc_data = xr.concat(list(floc_data.values()), dim="category")
    return floc_data.assign_coords(
        {
            coord: (coord, floc_data[coord].astype(str).data)
            for coord in ("category", "metric")
        }
        | {"superordinate": ("category", superordinate)},
    )
