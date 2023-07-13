from collections.abc import Sequence
import hashlib
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from bonner.files import download_from_s3
from bonner.datasets._utilities import nii
from bonner.datasets.allen2021_natural_scenes._utilities import (
    IDENTIFIER,
    BUCKET_NAME,
    CACHE_PATH,
    filter_betas_by_stimulus_id,
)
from bonner.datasets.allen2021_natural_scenes._stimuli import (
    load_stimulus_metadata,
)

RESOLUTION = "1pt8mm"
PREPROCESSING = "fithrf_GLMdenoise_RR"
N_SUBJECTS = 8
N_SESSIONS = (40, 40, 32, 30, 40, 32, 40, 30)
N_SESSIONS_HELD_OUT = 3
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


def _extract_stimulus_ids(subject: int) -> xr.DataArray:
    """Extract and format image IDs for all trials.

    Returns:
        stimulus_ids seen at each trial with "session" and "trial" dimensions
    """
    metadata = load_stimulus_metadata()
    metadata = np.array(
        metadata.loc[
            :, [f"subject{subject + 1}_" in column for column in metadata.columns]
        ]
    )
    assert metadata.shape[-1] == 3
    indices = np.nonzero(metadata)
    trials = metadata[indices] - 1  # fix 1-indexing

    stimulus_ids_ = [f"image{idx:05}" for idx in indices[0]]
    session_ids = trials // N_TRIALS_PER_SESSION
    intra_session_trial_ids = trials % N_TRIALS_PER_SESSION

    stimulus_ids = xr.DataArray(
        data=np.full((max(N_SESSIONS), N_TRIALS_PER_SESSION), "", dtype="<U10"),
        dims=("session", "trial"),
    )
    stimulus_ids.values[session_ids, intra_session_trial_ids] = stimulus_ids_
    return stimulus_ids.assign_coords(
        {dim: (dim, np.arange(stimulus_ids.sizes[dim])) for dim in stimulus_ids.dims}
    )


def load_brain_mask(*, subject: int, resolution: str) -> xr.DataArray:
    """Load and format a Boolean brain mask for the functional data.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
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


def load_validity(
    *, subject: int, resolution: str, exclude_held_out: bool = True
) -> xr.DataArray:
    validity = []
    n_sessions = N_SESSIONS[subject]
    if exclude_held_out:
        n_sessions -= N_SESSIONS_HELD_OUT

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
            .astype(dtype=bool, order="C")
        )
    return xr.concat(validity, dim="session")


def load_betas(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
    z_score: bool,
    neuroid_filter: Sequence[bool] | None = None,
    stimulus_filter: set[str] | None = None,
    exclude_held_out: bool = True,
) -> xr.DataArray:
    """Load betas.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"
        preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf"

    Returns:
        betas
    """
    stimulus_ids = _extract_stimulus_ids(subject)

    n_sessions = N_SESSIONS[subject]
    if exclude_held_out:
        n_sessions -= N_SESSIONS_HELD_OUT
    sessions = np.arange(n_sessions)

    validity = load_validity(subject=subject, resolution=resolution)
    validity = np.all(
        validity.stack({"neuroid": ("x", "y", "z")}, create_index=True).values[:-1, :],
        axis=0,
    )
    if neuroid_filter is not None:
        neuroid_filter = np.logical_and(neuroid_filter, validity)

    betas: list[xr.DataArray] | xr.DataArray = []

    run_id = []
    for i_run in range(N_RUNS_PER_SESSION):
        n_trials = 63 if i_run % 2 == 0 else 62
        run_id.extend([i_run] * n_trials)
    run_id = np.array(run_id).astype(np.uint8)

    for session in tqdm(sessions, desc="session"):
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
            xr.open_dataset(CACHE_PATH / filepath)["betas"]
            .rename(
                {
                    "phony_dim_0": "presentation",
                    "phony_dim_1": "z",
                    "phony_dim_2": "y",
                    "phony_dim_3": "x",
                }
            )
            .load()
            .transpose("x", "y", "z", "presentation")
            .astype(dtype=np.int16, order="C")
        )
        betas_session = (
            betas_session
            .assign_coords(
                {
                    dim: (dim, np.arange(betas_session.sizes[dim], dtype=np.uint8))
                    for dim in ("x", "y", "z")
                }
                | {
                    "stimulus_id": (
                        "presentation",
                        stimulus_ids.sel(session=session).data,
                    ),
                    "session_id": lambda x: (
                        "presentation",
                        session
                        * np.ones(x.sizes["presentation"], dtype=np.uint8),
                    ),
                    "trial": lambda x: (
                        "presentation",
                        np.arange(x.sizes["presentation"], dtype=np.uint16),
                    ),
                    "run_id": ("presentation", run_id)
                }
            )
            .stack({"neuroid": ("x", "y", "z")}, create_index=False)
        )
        if neuroid_filter is not None:
            betas_session = betas_session.isel({"neuroid": neuroid_filter})

        betas_session = (
            betas_session
            .transpose("neuroid", "presentation")
            .astype(dtype=np.float32, order="C")
            .transpose("presentation", "neuroid")
        )

        if z_score:
            betas_session = (betas_session - betas_session.mean("presentation")) / betas_session.std("presentation")
        else:
            betas_session /= 300

        if stimulus_filter is not None:
            betas_session = filter_betas_by_stimulus_id(
                betas=betas_session, stimulus_ids=stimulus_filter
            )

        betas.append(betas_session)

    betas = xr.concat(betas, dim="presentation").dropna(dim="neuroid", how="any")

    reps: dict[str, int] = {}
    rep_id: list[int] = []
    for stimulus_id in betas["stimulus_id"].values:
        if stimulus_id in reps:
            reps[stimulus_id] += 1
        else:
            reps[stimulus_id] = 0
        rep_id.append(reps[stimulus_id])
    rep_id = np.array(rep_id).astype(np.uint8)

    betas = (
        betas
        .assign_coords({"rep_id": ("presentation", rep_id)})
        .assign_attrs(
            {
                "subject": subject,
                "resolution": resolution,
                "preprocessing": preprocessing,
                "z_score": int(z_score),
                "presentations": _hash_index_coordinates(betas, coords=("session_id", "trial")),
                "neuroids": _hash_index_coordinates(betas, coords=("x", "y", "z")),
            }
        )
    )
    identifier = ".".join([f"{key}={value}" for key, value in betas.attrs.items()])
    return betas.rename(f"{IDENTIFIER}.{identifier}")


def _hash_index_coordinates(x: xr.DataArray, /, coords: Sequence[str]) -> str:
    coordinates = np.stack(
        [
            x[coord].values
            for coord in coords
        ],
        axis=-1,
    )
    coordinates = coordinates[np.lexsort(coordinates.transpose())]
    return hashlib.sha1(coordinates).hexdigest()


def load_ncsnr(
    *,
    subject: int,
    resolution: str,
    preprocessing: str,
) -> xr.DataArray:
    """Load and format noise-ceiling signal-to-noise ratios (NCSNR).

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"
        preprocessing: "fithrf_GLMdenoise_RR", "fithrf", or "assumehrf

    Returns:
        noise-ceiling SNRs
    """
    filepath = (
        Path("nsddata_betas")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / f"func{resolution}"
        / f"betas_{preprocessing}"
        / f"ncsnr.nii.gz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return nii.to_dataarray(CACHE_PATH / filepath).astype(dtype=np.float64, order="C")


def load_structural_scans(*, subject: int, resolution: str) -> xr.DataArray:
    """Load and format the structural scans registered to the functional data.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
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
            .astype(dtype=np.uint16, order="C")
        )
    return xr.concat(scans, dim="sequence").assign_coords(
        {"sequence": ("sequence", sequences)}
    )


def load_rois(
    *,
    subject: int,
    resolution: str,
) -> xr.DataArray:
    """Load the ROI masks for a subject.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
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
                filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath
            )

            mapping = (
                pd.read_csv(
                    CACHE_PATH / filepath,
                    delim_whitespace=True,
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
                    filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath
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
                            .astype(bool, order="C")
                        )
    rois = xr.concat(rois, dim="roi")
    rois["label"] = rois["roi"].astype(str)
    return rois.drop_vars("roi").set_index({"roi": ("source", "label", "hemisphere")})


def load_receptive_fields(*, subject: int, resolution: str) -> xr.DataArray:
    """Load population receptive field mapping data.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
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
        )
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
            .astype(dtype=np.float64, order="C")
        )
    return xr.concat(prf_data, dim="quantity").assign_coords(
        {"quantity": ("quantity", quantities)}
    )


def load_functional_contrasts(*, subject: int, resolution: str) -> xr.DataArray:
    """Load functional contrasts.

    Args:
        subject: subject ID
        resolution: "1pt8mm" or "1mm"

    Returns:
        functional contrasts
    """
    categories = {}
    for filename in ("domains", "categories"):
        filepath = Path("nsddata") / "experiments" / "floc" / f"{filename}.tsv"
        download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)

        categories[filename] = list(
            pd.read_csv(CACHE_PATH / filepath, sep="\t").iloc[:, 0].values
        )

    categories_combined = categories["domains"] + categories["categories"]
    superordinate = np.array(
        [
            True if category in categories["domains"] else False
            for category in categories_combined
        ],
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
                filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath
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
                .astype(np.float64, order="C")
            )
        floc_data[category] = xr.concat(floc_data[category], dim="metric")
    floc_data = xr.concat(list(floc_data.values()), dim="category")
    return floc_data.assign_coords(
        {
            coord: (coord, floc_data[coord].astype(str).values)
            for coord in ("category", "metric")
        }
        | {"superordinate": ("category", superordinate)}
    )
