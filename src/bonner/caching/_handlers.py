from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from collections.abc import Mapping
import pickle

from loguru import logger
from tqdm.dask import TqdmCallback
import numpy as np
import numpy.typing as npt
import xarray as xr


class Handler(ABC):
    def __init__(self) -> None:
        return

    @abstractmethod
    def save(self, result: Any, *, path: Path, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load(self, path: Path, **kwargs: Any) -> Any:
        pass


class NumpyHandler(Handler):
    def __init__(self) -> None:
        super().__init__()

    def save(
        self,
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
        self, path: Path, **kwargs: Any
    ) -> npt.NDArray[Any] | Mapping[str, npt.NDArray[Any]]:
        return np.load(path, **kwargs)


class XarrayHandler(Handler):
    def __init__(self) -> None:
        super().__init__()

    def save(
        self, result: xr.DataArray | xr.Dataset, *, path: Path, **kwargs: Any
    ) -> None:
        if isinstance(result, xr.DataArray) and result.size == 0:
            logger.warning(
                f"The result has size 0, writing an empty netCDF4 file to {path}"
            )
            xr.DataArray().to_netcdf(path)
        else:
            with TqdmCallback(desc="dask", leave=False):
                result.to_netcdf(path, **kwargs)

    def load(self, path: Path, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        try:
            return xr.open_dataarray(path, **kwargs)
        except Exception:
            return xr.open_dataset(path, **kwargs)


class PickleHandler(Handler):
    def __init__(self) -> None:
        super().__init__()

    def save(self, result: Any, *, path: Path, **kwargs) -> None:
        with open(path, "wb") as f:
            pickle.dump(result, f, **kwargs)

    def load(self, path: Path, **kwargs: Any) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f, **kwargs)


def get_handler(filetype: str) -> Handler:
    match filetype:
        case "numpy":
            return NumpyHandler()
        case "netCDF4":
            return XarrayHandler()
        case "pickle":
            return PickleHandler()
        case _:
            raise ValueError(f"Handler for filetype {filetype} not supported")
