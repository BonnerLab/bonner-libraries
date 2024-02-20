import itertools
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xr
from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_url, untar, unzip
from tqdm.auto import tqdm

IDENTIFIER = "hebart2023.things-data"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
URLS = {
    "meg.tar.gz": "https://plus.figshare.com/ndownloader/files/37650461",
    "fmri_betas.tar.gz": "https://plus.figshare.com/ndownloader/files/36806148",
    "brain_masks.zip": "https://plus.figshare.com/ndownloader/files/36682242",
    "rois.zip": "https://plus.figshare.com/ndownloader/files/38517326",
    "noise_ceilings.zip": "https://plus.figshare.com/ndownloader/files/36682266",
    "cortical_flatmaps.zip": "https://plus.figshare.com/ndownloader/files/36693528",
    "behavior.zip": "https://plus.figshare.com/ndownloader/files/38787423",
}

FUNCTIONAL_ROIS = {
    "body": ("EBA",),
    "face": ("FFA", "OFA", "STS"),
    "object": ("LOC",),
    "scene": ("PPA", "RSC", "TOS"),
}
ROIS = FUNCTIONAL_ROIS | {
    "pRF": (
        "V1",
        "V2",
        "V3",
        "hV4",
        "VO1",
        "VO2",
        "LO1",
        "LO2",
        "TO1",
        "TO2",
        "V3a",
        "V3b",
    ),
}

N_SUBJECTS = 3
N_SESSIONS = 12
N_RUNS_PER_SESSION = 10


def load_nii(filepath: Path) -> xr.DataArray:
    nii = nib.load(filepath).get_fdata()
    dims = ["x", "y", "z"]
    if nii.ndim == 4:
        dims.append("presentation")
    return xr.DataArray(
        data=nii,
        dims=dims,
        coords={
            dim: (dim, np.arange(nii.shape[i_dim], dtype=np.uint8))
            for i_dim, dim in enumerate(("x", "y", "z"))
        },
    ).stack({"neuroid": ("x", "y", "z")}, create_index=False)


def load_functional_rois(subject: int) -> xr.DataArray:
    def _download() -> Path:
        filename = "rois.zip"
        filepath = download_from_url(
            URLS[filename],
            filepath=CACHE_PATH / "downloads" / filename,
            force=False,
        )
        filepath = unzip(filepath, remove_zip=False, extract_dir=CACHE_PATH / "rois")
        return filepath

    def _package() -> xr.DataArray:
        masks = []
        for localizer_type, rois in FUNCTIONAL_ROIS.items():
            for roi, hemisphere in itertools.product(rois, ("l", "r")):
                path = (
                    CACHE_PATH
                    / "rois"
                    / "rois"
                    / "category_localizer"
                    / f"sub-{subject+1:02}"
                    / f"{localizer_type}_parcels"
                    / f"sub-{subject+1:02}_{hemisphere}{roi}.nii.gz"
                )
                try:
                    masks.append(
                        load_nii(path)
                        .astype(bool)
                        .expand_dims("roi")
                        .assign_coords(
                            {
                                "hemisphere": ("roi", [hemisphere]),
                                "localizer": ("roi", [localizer_type]),
                                "label": ("roi", [roi]),
                            },
                        ),
                    )
                except:
                    pass
        return xr.concat(masks, dim="roi")

    try:
        return _package()
    except:
        _download()
        return _package()


def load_receptive_fields(subject: int) -> xr.DataArray:
    def _download() -> Path:
        filename = "rois.zip"
        filepath = download_from_url(
            URLS[filename],
            filepath=CACHE_PATH / "downloads" / filename,
            force=False,
        )
        filepath = unzip(filepath, remove_zip=False, extract_dir=CACHE_PATH / "rois")
        return filepath

    def _package() -> xr.DataArray:
        quantities = {
            "angle": "angle",
            "eccentricity": "eccen",
            "sigma": "sigma",
            "roi": "varea",
        }
        prfs = []
        for quantity, label in quantities.items():
            path = (
                CACHE_PATH
                / "rois"
                / "rois"
                / "prf"
                / f"sub-{subject+1:02}"
                / f"resampled_{label}.nii.gz"
            )
            prfs.append(load_nii(path).expand_dims({"quantity": [quantity]}))
        return xr.concat(prfs, dim="quantity")

    try:
        return _package()
    except:
        _download()
        return _package()


def load_rois(subject: int) -> xr.DataArray:
    varea_mapping = {
        1: "V1",
        2: "V2",
        3: "V3",
        4: "hV4",
        5: "VO1",
        6: "VO2",
        7: "LO1",
        8: "LO2",
        9: "TO1",
        10: "TO2",
        11: "V3a",
        12: "V3b",
    }
    functional_rois = load_functional_rois(subject=subject)
    visual_rois = load_receptive_fields(subject=subject)
    rois = []
    for index, roi in varea_mapping.items():
        rois.append(
            (visual_rois.sel(quantity="roi") == index)
            .drop_vars("quantity")
            .expand_dims("roi")
            .assign_coords(
                {
                    "hemisphere": ("roi", [""]),
                    "localizer": ("roi", ["pRF"]),
                    "label": ("roi", [roi]),
                },
            ),
        )
    return xr.concat(rois + [functional_rois], dim="roi").set_index(
        {"roi": ["hemisphere", "localizer", "label"]},
    )


def load_brain_mask(subject: int) -> xr.DataArray:
    def _download() -> Path:
        filename = "brain_masks.zip"
        filepath = download_from_url(
            URLS[filename],
            filepath=CACHE_PATH / "downloads" / filename,
            force=False,
        )
        return unzip(
            filepath,
            remove_zip=False,
            extract_dir=CACHE_PATH / "brain_masks",
        )

    def _package() -> xr.DataArray:
        path = (
            CACHE_PATH
            / "brain_masks"
            / "brainmasks"
            / f"sub-{subject + 1:02}_space-T1w_brainmask.nii.gz"
        )
        return load_nii(path).astype(bool)

    try:
        return _package()
    except:
        _download()
        return _package()


def load_noise_ceilings(subject: int) -> xr.DataArray:
    def _download() -> Path:
        filename = "noise_ceilings.zip"
        filepath = download_from_url(
            URLS[filename],
            filepath=CACHE_PATH / "downloads" / filename,
            force=False,
        )
        return unzip(
            filepath,
            remove_zip=False,
            extract_dir=CACHE_PATH / "noise_ceilings",
        )

    def _package() -> xr.DataArray:
        noise_ceilings = []
        for n_images in range(12):
            path = (
                CACHE_PATH
                / "noise_ceilings"
                / "noise_ceilings"
                / f"sub-{subject+1:02}_nc_n-{n_images+1}.nii.gz"
            )
            noise_ceilings.append(load_nii(path).expand_dims({"n_images": [n_images]}))
        return xr.concat(noise_ceilings, dim="n_images")

    try:
        return _package()
    except:
        _download()
        return _package()


def load_betas(
    *,
    subject: int,
    neuroid_filter: Sequence[bool] | None = None,
) -> xr.DataArray:
    def _download() -> Path:
        filename = "fmri_betas.tar.gz"
        filepath = download_from_url(
            URLS[filename],
            filepath=CACHE_PATH / "downloads" / filename,
            force=False,
        )
        return untar(
            filepath,
            remove_tar=False,
            extract_dir=CACHE_PATH / "fmri_betas",
        )

    def _package() -> xr.DataArray:
        betas = []
        for session in tqdm(range(N_SESSIONS), desc="session", leave=False):
            for run in tqdm(range(N_RUNS_PER_SESSION), desc="run", leave=False):
                path_stem = (
                    CACHE_PATH
                    / "fmri_betas"
                    / "scalematched"
                    / f"sub-{subject + 1:02}"
                    / f"ses-things{session + 1:02}"
                    / f"sub-{subject + 1:02}_ses-things{session + 1:02}_run-{run + 1:02}"
                )
                conditions = pd.read_csv(
                    path_stem.with_name(f"{path_stem.name}_conditions.tsv"),
                    sep="\t",
                    header=0,
                    index_col=0,
                )
                betas_session = load_nii(
                    path_stem.with_name(f"{path_stem.name}_betas.nii.gz"),
                )
                betas.append(
                    betas_session.assign_coords(
                        {
                            "stimulus": (
                                "presentation",
                                [
                                    Path(filename).stem
                                    for filename in conditions["image_filename"]
                                ],
                            ),
                            "session": (
                                "presentation",
                                session
                                * np.ones(
                                    betas_session.sizes["presentation"],
                                    dtype=np.uint8,
                                ),
                            ),
                            "run": (
                                "presentation",
                                run
                                * np.ones(
                                    betas_session.sizes["presentation"],
                                    dtype=np.uint8,
                                ),
                            ),
                        },
                    )
                    .isel({"neuroid": neuroid_filter})
                    .transpose("presentation", "neuroid")
                    .astype(dtype=np.float32),
                )

        betas = xr.concat(betas, dim="presentation")

        # exclude catch trials labelled catchNNN_<something>
        betas = betas.isel(
            {
                "presentation": [
                    stimulus.split("_")[0][:5] != "catch"
                    for stimulus in betas["stimulus"].data
                ],
            },
        )

        reps: dict[str, int] = {}
        repetitions = np.empty(
            betas.sizes["presentation"],
            dtype=np.uint8,
        )
        for i_stimulus, stimulus in enumerate(betas["stimulus"].data):
            if stimulus in reps:
                reps[stimulus] += 1
            else:
                reps[stimulus] = 0
            repetitions[i_stimulus] = reps[stimulus]

        return betas.assign_coords({"repetition": ("presentation", repetitions)})

    try:
        return _package()
    except:
        _download()
        return _package()
