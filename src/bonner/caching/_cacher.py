__all__ = ["Cacher"]

from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable, Mapping
from functools import wraps
import inspect
from pathlib import Path
import os

import numpy as np
import xarray as xr

from bonner.caching._handlers import get_handler

P = ParamSpec("P")
R = TypeVar("R")


class Cacher:
    def __init__(  # type: ignore  # kwargs can be Any
        self,
        identifier: str = None,
        *,
        path: Path = Path(
            os.getenv(
                "BONNER_CACHING_CACHE", str(Path.home() / ".cache" / "bonner-caching")
            )
        ),
        helper: Callable[[Mapping[str, Any]], dict[str, str]] = None,
        filetype: str = "auto",
        mode: str = os.getenv("BONNER_CACHING_MODE", "normal"),
        kwargs_save: Mapping[str, Any] = {},
        kwargs_load: Mapping[str, Any] = {},
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Caches outputs of functions to disk so that the (potentially expensive) function is not re-evaluated when called again.

        When the cacher is called on a function, it computes the output of the function and stores it on disk at the path ``path / identifier``. If the function is called again, the cached value is retrieved from disk and returned.

        The identifier can be parametrized by the function inputs: for example, if the function takes in the integer x as input, setting the identifier to "{x}.pkl" will result in the filename "2.pkl". This is accomplished by calling ``identifier.format(*args, **kwargs)``, which requires that the template arguments have direct string representations. For additional flexibility, the cacher offers the ``helper`` argument, which must be a function that takes in the arguments to the function as a dictionary and returns a dictionary mapping template variables to actual values, including potential evaluations.

        Basic usage:

        The following example will cache the output of ``add(3, 5)`` to ``~/output/sums/first_arg_3/second_arg_5.pkl`` as a Python pickle file.

        ```
        from pathlib import Path

        @cache(path=Path.home() / "output", identifier="sums/first_arg_{x}/second_arg_{y}.pkl", filetype="pickle")
        def add(x: int, y: int) -> int:
            return x + y
        ```

        Advanced usage:

        The following example will cache the output of ``add({"three": 3, "five": 5})`` ``to $BONNER_CACHING_CACHE/analysis/keys=three.five/values=3_5/True.pkl``.

        ```
        analysis = "fancy_sum"

        @cache(
            identifier=f"{analysis}/keys={{dict_keys}}/values={{dict_values}}/{{flag}}.pkl",
            helper=lambda kwargs: {
                "dict_keys": ".".join(list(kwargs["x"].keys())),
                "dict_values": "_".join(list(kwargs["x"].values())),
                "flag": kwargs["flag"],
            }
        )
        def add(x: dict[str, float], flag: bool = True) -> float:
            return sum(list(x.values()))
        ```

        Todo:
            * Support filetype = "auto", which auto-detects filetype based on output class
            * Track progress of :PEP: `501` (https://peps.python.org/pep-0501/) which introduces lazy f-strings. This would allow for a simpler implementation without the ``helper`` argument.

        Args:
            path: Cache directory to save to/load from. Defaults to the value of the environment variable BONNER_CACHING_CACHE. If the environment variable is not set, defaults to ``~/.cache/bonner-caching``.
            mode: Controls the behavior of the cacher:
                * "normal": If the function output has been previously cached, the stored value is retrieved and returned. If the output has not been previously cached, the function body is run and the output is cached.
                * "readonly": If the function output has been previously cached, the stored value is retrieved and returned. If the output has not been previously cached, the function body is run but the output is NOT cached.
                * "overwrite": The function body is run and the output is cached, overwriting any existing cached output. Existing cached values are not read from.
                * "delete": The function body is run and the output is returned. Any existing cached values are deleted.
                * "ignore": The function body is run and the output is returned. Any existing cached values are ignored.
                Defaults to the value of the environment variable BONNER_CACHING_MODE. If the environment variable is not set, defaults to "normal".
            identifier: _description_. Defaults to None.
            filetype: Serialization protocol used to cache the files to disk. Supported filetypes include:
                * "auto" (default): Uses one of the following protocols depending on the function output. Requires that the suffix of the 'identifier' is '.npy' for numpy, '.nc' for netCDF4, and '.pkl' for pickle.
                * "numpy": If the function output is a single numpy array, the `numpy.save` function is used.
                * "netCDF4": If the function output is an xarray.DataArray or an xarray.Dataset, the `.to_netcdf` method is used to save the variables to disk.
                * "pickle": All other function outputs are pickled.
                 Defaults to "auto".
            kwargs_save: Keyword arguments passed on to the `.save` method of the `Handler` corresponding to `filetype`. Defaults to {}.
            kwargs_load: Keyword arguments passed on to the `.load` method of the `Handler` corresponding to `filetype`. Defaults to {}.

        Returns:
            Any: The output of the function.
        """
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        modes = {"normal", "readonly", "overwrite", "delete", "ignore"}
        assert mode in modes, f"mode {mode} not supported (allowed modes: {modes}"

        self.identifier = identifier
        self.helper = helper
        self.filetype = filetype
        self.kwargs_save = kwargs_save
        self.kwargs_load = kwargs_load

    def __call__(
        self,
        func: Callable[P, R],
    ) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            args_to_format = self._get_args(func, *args, **kwargs)

            if self.helper is not None:
                args_to_format = self.helper(args_to_format)
            identifier = self.identifier.format(**args_to_format)

            match self.mode:
                case "normal":
                    if self._get_path(identifier):
                        result = self._load(identifier, **self.kwargs_load)
                    else:
                        result = func(*args, **kwargs)
                        self._save(result, identifier=identifier, **self.kwargs_save)
                case "readonly":
                    if self._get_path(identifier):
                        result = self._load(identifier, **self.kwargs_load)
                    else:
                        result = func(*args, **kwargs)
                case "overwrite":
                    result = func(*args, **kwargs)
                    self._save(result, identifier=identifier, **self.kwargs_save)
                case "delete":
                    if self._get_path(identifier):
                        self._delete(identifier)
                    result = func(*args, **kwargs)
                case "ignore":
                    result = func(*args, **kwargs)
                case _:
                    raise ValueError(
                        f"mode must be one of 'normal', 'readonly', 'overwrite',"
                        f" 'delete', or 'ignore'"
                    )
            return result

        return wrapper

    def _save(self, result: Any, *, identifier: str) -> None:  # type: ignore  # result can be Any
        path = self.path / identifier
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.filetype == "auto":
            if isinstance(result, np.ndarray):
                filetype = "numpy"
                if path.suffix != ".npy":
                    raise ValueError(
                        "identifier must have suffix '.npy' if filetype is 'auto'"
                    )
            elif isinstance(result, (xr.DataArray, xr.Dataset)):
                filetype = "netCDF4"
                if path.suffix != ".nc":
                    raise ValueError(
                        "identifier must have suffix '.nc' if filetype is 'auto'"
                    )
            else:
                filetype = "pickle"
                if path.suffix != ".pkl":
                    raise ValueError(
                        "identifier must have suffix '.pkl' if filetype is 'auto'"
                    )
        else:
            filetype = self.filetype

        handler = get_handler(filetype=filetype)
        handler.save(result=result, path=path, **self.kwargs_save)

    def _load(self, identifier: str) -> Any:  # type: ignore  # file contents can be Any
        path = self._get_path(identifier)

        if self.filetype == "auto":
            match path.suffix:
                case ".npy":
                    filetype = "numpy"
                case ".nc":
                    filetype = "netCDF4"
                case ".pkl":
                    filetype = "pickle"
        else:
            filetype = self.filetype

        handler = get_handler(filetype=filetype)
        return handler.load(path=path, **self.kwargs_load)

    def _delete(self, identifier: str) -> None:
        filepath = self._get_path(identifier)
        filepath.unlink()

    def _get_path(self, identifier: str) -> Path | None:
        path = self.path / identifier
        if path.exists():
            return path
        else:
            return None

    def _get_args(  # type: ignore  # arguments can be Any
        self, function: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> dict[str, Any]:
        signature = inspect.signature(function)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments.arguments
