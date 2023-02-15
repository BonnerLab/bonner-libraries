"""
Adapted from https://github.com/cvnlab/nsdcode
"""

from collections.abc import Collection
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import nibabel as nib

from bonner.files import download_from_s3
from bonner.datasets.allen2021_natural_scenes._utilities import BUCKET_NAME, CACHE_PATH

MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1


def load_transformation(
    subject: int, *, source_space: str, target_space: str, suffix: str
) -> np.ndarray:
    filepath = (
        Path("nsddata")
        / "ppdata"
        / f"subj{subject + 1:02}"
        / "transforms"
        / f"{source_space}-to-{target_space}{suffix}"
    )

    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    transformation = nib.load(CACHE_PATH / filepath).get_fdata()
    return transformation


def load_native_surface(
    subject: int, *, hemisphere: str, surface_type: str = "w-g.pct.mgh"
) -> Path:
    filepath = (
        Path("nsddata")
        / "freesurfer"
        / f"subj{subject + 1:02}"
        / f"surf"
        / f"{hemisphere}.{surface_type}"
    )
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return CACHE_PATH / filepath


def _interpolate(
    volume: np.ndarray, *, coordinates: np.ndarray, interpolation_type: str = "cubic"
) -> np.ndarray:
    """
    Wrapper for ba_interp3. Normal calls to ba_interp3 assign values to interpolation points that lie outside the original data range. We ensure that coordinates outside the original field-of-view (i.e. if the value along a dimension is less than 1 or greater than the number of voxels in the original volume along that dimension) are returned as NaN and coordinates that have any NaNs are returned as NaN.

    Args:
        volume: 3D matrix (can be complex-valued)
        coordinates: (3, N) matrix coordinates to interpolate at
        interpolation_type: "nearest", "linear", or "cubic"
    """
    # input
    match interpolation_type:
        case "cubic":
            order = 3
        case "linear":
            order = 1
        case "nearest":
            order = 0
        case _:
            raise ValueError("interpolation method not implemented")

    # bad locations must get set to NaN
    bad = np.any(np.isinf(coordinates), axis=0)
    coordinates[:, bad] = -1

    # out of range must become NaN, too
    bad = np.any(
        np.c_[
            bad,
            coordinates[0, :] < 0,
            coordinates[0, :] > volume.shape[0] - 1,
            coordinates[1, :] < 0,
            coordinates[1, :] > volume.shape[1] - 1,
            coordinates[2, :] < 0,
            coordinates[2, :] > volume.shape[2] - 1,
        ],
        axis=1,
    ).astype(bool)

    transformed_data = map_coordinates(
        np.nan_to_num(volume).astype(np.float64),
        coordinates,
        order=order,
        mode="nearest",
    )
    transformed_data[bad] = np.nan

    return transformed_data


def _transform(
    data: np.ndarray,
    *,
    transformation: np.ndarray,
    interpolation_type: str,
    target_type: str,
) -> np.ndarray:
    """_summary_

    Args:
        data: data to be transformed from one space to another
        transformation: transformation matrix
        interpolation_type: passed to _interpolate
        target_type: "volume" or "surface"

    Returns:
        Transformed data
    """
    target_shape = transformation.shape[:3]

    coordinates = np.c_[
        transformation[..., 0].ravel(order="F"),
        transformation[..., 1].ravel(order="F"),
        transformation[..., 2].ravel(order="F"),
    ].T

    coordinates -= 1  # Kendrick's 1-based indexing.

    data_ = _interpolate(
        data, coordinates=coordinates, interpolation_type=interpolation_type
    )
    data_ = np.nan_to_num(data_)
    if target_type == "volume":
        data_ = data_.reshape(target_shape, order="F")

    return data_


def convert_ndarray_to_nifti1image(
    data: np.ndarray,
    *,
    resolution: float = MNI_RESOLUTION,
    origin: np.ndarray = MNI_ORIGIN,
) -> nib.Nifti1Image:
    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return nib.Nifti1Image(data, affine, header)


def transform_volume_to_mni(
    data: np.ndarray, *, subject: int, source_space: str, interpolation_type: str
) -> np.ndarray:
    transformation = load_transformation(
        subject=subject, source_space=source_space, target_space="MNI", suffix=".nii.gz"
    )
    transformed_data = _transform(
        data=data,
        transformation=transformation,
        target_type="volume",
        interpolation_type=interpolation_type,
    )
    return transformed_data


def transform_volume_to_native_surface(
    data: np.ndarray,
    *,
    subject: int,
    source_space: str = "func1pt8",
    interpolation_type: str = "cubic",
    layers: Collection[str] = (
        "layerB1",
        "layerB2",
        "layerB3",
    ),
    average_across_layers: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    native_surface: dict[str, dict[str, np.ndarray]] = {}
    for hemisphere in ("lh", "rh"):
        native_surface[hemisphere] = {}
        for layer in layers:
            transformation = load_transformation(
                subject=subject,
                source_space=f"{hemisphere}.{source_space}",
                target_space=layer,
                suffix=".mgz",
            )

            native_surface[hemisphere][layer] = _transform(
                data,
                transformation=transformation,
                target_type="surface",
                interpolation_type=interpolation_type,
            )

        if average_across_layers:
            native_surface[hemisphere] = {
                "average": np.vstack(list(native_surface[hemisphere].values())).mean(
                    axis=0
                )
            }
    return native_surface
