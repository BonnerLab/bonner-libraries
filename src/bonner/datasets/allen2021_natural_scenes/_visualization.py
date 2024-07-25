"""Adapted from https://github.com/cvnlab/nsdcode."""

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import xarray as xr
from bonner.datasets.allen2021_natural_scenes import load_brain_mask
from bonner.datasets.allen2021_natural_scenes._utilities import BUCKET_NAME, CACHE_PATH
from bonner.files import download_from_s3
from bonner.plotting._nilearn import _normalize_curv_map
from matplotlib.axes import Axes
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf
from scipy.ndimage import map_coordinates

MNI_SHAPE = (182, 218, 182)
MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1


def normalize_hemisphere(
    hemisphere: Literal["left", "right", "lh", "rh", "l", "r"],
) -> str:
    match hemisphere.lower():
        case "left" | "l" | "lh":
            return "lh"
        case "right" | "r" | "rh":
            return "rh"
        case _:
            raise ValueError


def load_transformation(
    *,
    subject: int,
    source_space: Literal["func1pt8", "lh.func1pt8", "rh.func1pt8"],
    target_space: Literal["MNI", "layerB1", "layerB2", "layerB3"],
    suffix: Literal[".nii.gz", ".mgz"],
) -> np.ndarray:
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "transforms"
        / f"{source_space}-to-{target_space}{suffix}"
    )

    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return nib.loadsave.load(CACHE_PATH / filepath).get_fdata()


def load_surface_roi(
    *,
    subject: int,
    hemisphere: Literal["left", "right"],
    label: Literal["nsdgeneral", "prf-visualrois", "streams"],
) -> Path:
    """Load and format a surface ROI."""
    filepath = (
        Path("nsddata")
        / "freesurfer"
        / f"subj{subject + 1:02}"
        / "label"
        / f"{normalize_hemisphere(hemisphere)}.{label}.mgz"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return CACHE_PATH / filepath


def load_surface_mesh(
    *,
    subject: int,
    hemisphere: Literal["left", "right"],
    label: Literal["inflated", "pial", "curv", "w-g.pct.mgh"],
) -> Path:
    """Load and format a surface mesh."""
    filepath = (
        Path("nsddata")
        / "freesurfer"
        / f"subj{subject + 1:02}"
        / "surf"
        / f"{normalize_hemisphere(hemisphere)}.{label}"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return CACHE_PATH / filepath


def convert_ndarray_to_nifti1image(
    data: np.ndarray,
    *,
    resolution: float = MNI_RESOLUTION,
    origin: np.ndarray = MNI_ORIGIN,
) -> nib.nifti1.Nifti1Image:
    header = nib.nifti1.Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return nib.nifti1.Nifti1Image(data, affine, header)


def _postprocess_surface_transform(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, xr.DataArray]:
    coordinates = np.squeeze(coordinates).transpose()  # flatten 3D to 1D
    coordinates -= 1  # compensate for 1-indexing
    good_voxels = xr.DataArray(
        data=np.all(np.isfinite(coordinates), axis=0),
        dims=("neuroid",),
    )
    return coordinates, good_voxels


def _postprocess_mni_transform(
    coordinates: np.ndarray,
    *,
    volume: np.ndarray,
) -> tuple[np.ndarray, xr.DataArray]:
    coordinates = np.flip(coordinates, axis=0)  # RPI to LPI ordering
    coordinates -= 1  # compensate for 1-indexing

    good_voxels = np.all(
        np.stack(
            [
                (coordinates[..., dim] >= 0)
                & (coordinates[..., dim] < volume.shape[dim])
                for dim in (-3, -2, -1)
            ]
            + [
                np.all(np.isfinite(coordinates), axis=-1),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    good_voxels = xr.DataArray(
        data=good_voxels,
        dims=("x", "y", "z"),
    ).stack({"neuroid": ("x", "y", "z")})

    coordinates = coordinates.reshape(-1, 3).transpose()  # flatten 3D to 1D

    return coordinates, good_voxels


def transform_data_to_mni(
    data: xr.DataArray,
    *,
    subject: int,
    source_space: Literal["func1pt8"] = "func1pt8",
    order: int = 0,
) -> xr.DataArray:
    brain_shape = load_brain_mask(subject=subject, resolution="1pt8mm").shape
    volume = reshape_dataarray_to_brain(data, brain_shape=brain_shape)

    coordinates = load_transformation(
        subject=subject,
        source_space=source_space,
        target_space="MNI",
        suffix=".nii.gz",
    )
    shape = coordinates.shape[:-1]
    coordinates, good_voxels = _postprocess_mni_transform(
        coordinates,
        volume=volume,
    )
    return xr.DataArray(
        data=map_coordinates(
            np.nan_to_num(volume.astype(np.float64), nan=0),
            coordinates[..., good_voxels],
            order=order,
            mode="nearest",
            output=np.float32,
        ),
        dims=("neuroid",),
        coords=good_voxels[good_voxels].coords,
        attrs={"shape": shape},
    )


def transform_data_to_surface(
    data: xr.DataArray,
    *,
    subject: int,
    source_space: Literal["func1pt8"] = "func1pt8",
    order: int,
) -> dict[str, dict[str, np.ndarray]]:
    brain_shape = load_brain_mask(subject=subject, resolution="1pt8mm").shape
    volume = reshape_dataarray_to_brain(data, brain_shape=brain_shape)

    surface: dict[str, dict[str, np.ndarray]] = {}
    for hemisphere in ("left", "right"):
        surface[hemisphere] = {}
        for layer in ("layerB1", "layerB2", "layerB3"):
            coordinates = load_transformation(
                subject=subject,
                source_space=f"{normalize_hemisphere(hemisphere)}.{source_space}",
                target_space=layer,
                suffix=".mgz",
            )
            coordinates, good_voxels = _postprocess_surface_transform(coordinates)

            surface[hemisphere][layer] = map_coordinates(
                np.nan_to_num(volume.astype(np.float64), nan=0),
                coordinates,
                order=order,
                mode="nearest",
                output=np.float32,
            )
            surface[hemisphere][layer][~good_voxels] = np.nan

        surface[hemisphere]["average"] = np.vstack(
            list(surface[hemisphere].values()),
        ).mean(axis=0)
    return surface


def plot_brain_map(
    data: xr.DataArray,
    *,
    ax: Axes,
    subject: int,
    space: Literal["surface", "MNI"] = "native",
    hemisphere: Literal["left", "right"] = "left",
    surface_type: Literal["pial", "inflated"] = "inflated",
    view: str | tuple[float, float] = "lateral",
    cmap: str = "cold_hot",
    interpolation: Literal["nearest", "linear", "cubic"] = "nearest",
    layer: Literal["layerB1", "layerB2", "layerB3", "average"] = "average",
    fsaverage_mesh: Literal["fsaverage"] = "fsaverage",
    **kwargs,
) -> None:
    match interpolation:
        case "nearest":
            order = 0
        case "linear":
            order = 1
        case "cubic":
            order = 3
        case _:
            raise ValueError

    match space:
        case "surface":
            stat_map = transform_data_to_surface(
                data,
                subject=subject,
                order=order,
            )[hemisphere][layer]

            curv_map = load_surf_data(
                load_surface_mesh(
                    subject=subject,
                    hemisphere=hemisphere,
                    label="curv",
                ),
            )
            surf_mesh = load_surface_mesh(
                subject=subject,
                hemisphere=hemisphere,
                label=surface_type,
            )
        case "MNI":
            mni = transform_data_to_mni(
                data,
                subject=subject,
                order=order,
            )
            mni = reshape_dataarray_to_brain(mni, brain_shape=MNI_SHAPE)
            fsaverage = fetch_surf_fsaverage(mesh=fsaverage_mesh)

            match surface_type:
                case "inflated":
                    prefix = "infl"
                case "pial":
                    prefix = "pial"
                case _:
                    raise ValueError

            surf_mesh = load_surf_mesh(fsaverage[f"{prefix}_{hemisphere}"])
            stat_map = vol_to_surf(
                convert_ndarray_to_nifti1image(mni),
                fsaverage[f"pial_{hemisphere}"],
            )
            curv_map = load_surf_data(fsaverage[f"curv_{hemisphere}"])
        case _:
            raise ValueError

    _ = plot_surf_stat_map(
        axes=ax,
        stat_map=stat_map,
        surf_mesh=surf_mesh,
        hemi=hemisphere,
        threshold=np.finfo(np.float32).resolution,
        colorbar=False,
        bg_map=_normalize_curv_map(
            curv_map,
            low=0.25,
            high=0.5,
        ),
        engine="matplotlib",
        view=view,
        cmap=cmap,
        **kwargs,
    )


def reshape_dataarray_to_brain(
    data: xr.DataArray,
    *,
    brain_shape: tuple[int, ...],
) -> np.ndarray:
    output_shape = (
        (data.sizes["presentation"], *brain_shape) if data.ndim == 2 else brain_shape
    )
    output = np.full(output_shape, fill_value=np.nan, dtype=data.data.dtype)
    output[..., data["x"].data, data["y"].data, data["z"].data] = data.data
    return output
