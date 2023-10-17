from pathlib import Path
import numpy as np
import xarray as xr

from bonner.datasets._utils import load_nii
from bonner.datasets.chang2019_bold5000._utilities import (
    IDENTIFIER,
    N_SESSIONS,
    ROIS,
    get_betas_filename,
    get_brain_mask_filename,
    get_imagenames_filename,
)


def create_assembly(subject: int) -> xr.DataArray:
    mask = load_brain_mask(subject)
    structural_scan = load_structural_scan(subject)
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
            }
        )
        .assign_coords(
            {coord: neuroid_metadata[coord] for coord in neuroid_metadata.coords}
        )
        .assign_attrs(
            {
                "brain_dimensions": mask.attrs["brain_dimensions"],
                "structural_scan": structural_scan.data,
                "structural_scan_brain_dimensions": structural_scan.attrs[
                    "brain_dimensions"
                ],
                "identifier": f"{IDENTIFIER}-subject{subject}",
                "stimulus_set_identifier": IDENTIFIER,
            }
        )
        .dropna(dim="neuroid", how="any")
    )


def load_neuroid_metadata(subject: int) -> xr.DataArray:
    brain_mask = load_brain_mask(subject)
    n_voxels = np.sum(brain_mask.data)
    metadata = xr.DataArray(np.full(n_voxels, np.nan), dims="neuroid").assign_coords(
        {"hemisphere": ("neuroid", [""] * n_voxels)}
    )

    for roi in ROIS:
        metadata = metadata.assign_coords(
            {f"roi_{roi}": ("neuroid", [False] * n_voxels)}
        )
        for hemisphere in ("LH", "RH"):
            roi_mask = load_roi_mask(subject, hemisphere, roi).sel(
                {"neuroid": brain_mask}
            )
            metadata[f"roi_{roi}"][roi_mask] = True
            metadata["hemisphere"][roi_mask] = hemisphere
    return metadata


def load_brain_mask(subject: int) -> xr.DataArray:
    return load_nii(Path.cwd() / get_brain_mask_filename(subject)).astype(bool)


def load_activations(subject: int, session: int) -> xr.DataArray:
    return load_nii(Path.cwd() / get_betas_filename(subject, session)).astype(
        np.float32
    )


def load_roi_mask(subject: int, hemisphere: str, roi: str) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"sub-CSI{subject + 1}/sub-CSI{subject + 1}_mask-{hemisphere}{roi}.nii.gz"
    ).astype(bool)


def load_structural_scan(subject: int) -> xr.DataArray:
    return load_nii(
        Path.cwd()
        / f"BOLD5000_Structural/CSI{subject + 1}_Structural/T1w_MPRAGE_CSI{subject + 1}.nii"
    )


def load_image_filename_stems(subject: int) -> list[Path]:
    with open(get_imagenames_filename(subject), "r") as f:
        # strip newlines, extension
        return [Path(line[:-1]).stem for line in f.readlines()]
