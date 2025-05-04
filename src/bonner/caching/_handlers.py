import contextlib
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import cf_xarray
import nibabel as nib
import numpy as np
import numpy.typing as npt
import xarray as xr
from PIL import Image


class Handler(ABC):
    def __init__(self: Self) -> None:
        return

    @abstractmethod
    def save(self: Self, result: Any, *, path: Path, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load(self: Self, path: Path, **kwargs: Any) -> Any:
        pass


class NumpyHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    def save(
        self: Self,
        result: npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]],
        *,
        path: Path,
        compress: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(result, np.ndarray):
            np.save(path, result, **kwargs)
        elif isinstance(result, dict):
            if compress:
                np.savez_compressed(path, **result)
            else:
                np.savez(path, **result)

    def load(
        self: Self,
        path: Path,
        **kwargs: Any,
    ) -> npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]]:
        return np.load(path, **kwargs)


class XarrayHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    def save(
        self: Self,
        result: xr.DataArray | xr.Dataset,
        *,
        path: Path,
        **kwargs: Any,
    ) -> None:
        if isinstance(result, xr.DataArray):
            result = result.to_dataset()

        # if the Dataset has multi-indexes, serialize them
        with contextlib.suppress(ValueError):
            result = cf_xarray.encode_multi_index_as_compress(result)

        result.to_netcdf(path, **kwargs)

    def load(self: Self, path: Path, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        result = xr.open_dataset(path, **kwargs)

        # if the Dataset has serialized multi-indexes, deserialize them
        with contextlib.suppress(ValueError):
            result = cf_xarray.decode_compress_to_multi_index(result)

        # if the Dataset has a single variable, convert it to a DataArray
        if len(result.data_vars) == 1:
            return result[next(iter(result.keys()))]

        return result


class PickleHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    def save(self: Self, result: Any, *, path: Path, **kwargs) -> None:
        with path.open("wb") as f:
            pickle.dump(result, f, **kwargs)

    def load(self: Self, path: Path, **kwargs: Any) -> Any:
        with path.open("rb") as f:
            return pickle.load(f, **kwargs)


class Nifti1ImageHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    def save(
        self: Self,
        result: nib.nifti1.Nifti1Image,
        *,
        path: Path,
        **kwargs: Any,
    ) -> None:
        if isinstance(result, nib.nifti1.Nifti1Image):
            nib.save(result, path)

    def load(self: Self, path: Path, **kwargs: Any) -> nib.nifti1.Nifti1Image:
        return nib.load(path)


class PillowImageHandler(Handler):
    def __init__(self: Self) -> None:
        super().__init__()

    def save(self: Self, result: Image.Image, *, path: Path, **kwargs: Any) -> None:
        result.save(path, **kwargs)

    def load(self: Self, path: Path, **kwargs: Any) -> Image.Image:
        return Image.open(path, **kwargs)


def get_handler(filetype: str) -> Handler:
    match filetype:
        case "numpy":
            return NumpyHandler()
        case "netCDF4":
            return XarrayHandler()
        case "pickle":
            return PickleHandler()
        case "NIfTI":
            return Nifti1ImageHandler()
        case "PIL":
            return PillowImageHandler()
        case _:
            error = f"Handler for filetype {filetype} not supported"
            raise ValueError(error)
