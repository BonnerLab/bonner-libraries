from pathlib import Path

import numpy as np
import xarray as xr

from bonner.datasets._utilities import nii
from bonner.datasets.chang2019_bold5000._utilities import (
    IDENTIFIER,
    N_SESSIONS,
    ROIS,
    get_betas_filename,
    get_brain_mask_filename,
    get_imagenames_filename,
)

from ._download import download_dataset


def load_betas(subject: int) -> xr.DataArray:
    download_dataset()

    mask = load_brain_mask(subject)
    neuroid_metadata = load_neuroid_metadata(subject)

    return (
        xr.concat(
            [
                load_activations(subject, session).sel({"neuroid": mask})
                for session in range(N_SESSIONS[subject])
            ],
            dim="presentation",
        )
        .astype(np.float32)
        .rename(f"{IDENTIFIER}-subject{subject}")
        .assign_coords(
            {
                "stimulus": (
                    "presentation",
                    load_image_filename_stems(subject),
                ),
            },
        )
        .assign_coords(
            {coord: neuroid_metadata[coord] for coord in neuroid_metadata.coords},
        )
        .dropna(dim="neuroid", how="any")
    )


def load_neuroid_metadata(subject: int) -> xr.DataArray:
    brain_mask = load_brain_mask(subject)
    n_voxels = np.sum(brain_mask.data)
    metadata = xr.DataArray(np.full(n_voxels, np.nan), dims="neuroid").assign_coords(
        {"hemisphere": ("neuroid", [""] * n_voxels)},
    )

    for roi in ROIS:
        metadata = metadata.assign_coords(
            {f"roi_{roi}": ("neuroid", [False] * n_voxels)},
        )
        for hemisphere in ("LH", "RH"):
            roi_mask = load_roi_mask(subject, hemisphere, roi).sel(
                {"neuroid": brain_mask},
            )
            metadata[f"roi_{roi}"][roi_mask] = True
            metadata["hemisphere"][roi_mask] = hemisphere
    return metadata


def load_brain_mask(subject: int) -> xr.DataArray:
    return nii.to_dataarray(Path.cwd() / get_brain_mask_filename(subject)).astype(bool)


def load_activations(subject: int, session: int) -> xr.DataArray:
    return nii.to_dataarray(Path.cwd() / get_betas_filename(subject, session)).astype(
        np.float32,
    )


def load_roi_mask(subject: int, hemisphere: str, roi: str) -> xr.DataArray:
    return nii.to_dataarray(
        Path.cwd()
        / f"sub-CSI{subject + 1}"
        / f"sub-CSI{subject + 1}_mask-{hemisphere}{roi}.nii.gz",
    ).astype(bool)


def load_structural_scan(subject: int) -> xr.DataArray:
    return nii.to_dataarray(
        Path.cwd() / "BOLD5000_Structural"
        f"/CSI{subject + 1}_Structural"
        f"/T1w_MPRAGE_CSI{subject + 1}.nii",
    )


def load_image_filename_stems(subject: int) -> list[str]:
    with get_imagenames_filename(subject).open("rb") as f:
        # strip newlines, extension
        return [Path(line[:-1]).stem for line in f.readlines()]
