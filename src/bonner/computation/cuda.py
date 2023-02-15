from typing import ParamSpec, TypeVar, Any
from collections.abc import Callable, Collection, Mapping
from functools import wraps
import gc

from loguru import logger
import torch

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_DEVICES: list[torch.device | None] = [
    torch.device(f"cuda:{gpu}") for gpu in range(torch.cuda.device_count())
] + [torch.device("cpu")]


class Environment:
    def __init__(self, environments: Collection[Mapping[str, Any]]) -> None:
        self.environments = environments

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        """Try running a function with various values of specified kwargs.

        Attempts to runs the function `func` with each set of kwargs specified by `environments`.

        Example:

        ```
        import numpy as np
        import numpy.typing as npy
        import torch

        n = 10
        x = np.zeros((n, n), dtype=np.float32)
        y = np.ones((n, n), dtype=np.float32)

        def add(x: npt.NDArray[np.float32], *, y: npt.NDArray[np.float32], device: torch.device = None) -> npt.NDArray[np.float32]:
            if device == torch.device("cuda:0"):
                return (torch.from_numpy(x).to(device) + torch.from_numpy(y).to(device)).cpu().numpy()
            else:
                return x + y

        z = try_environments(add, environments=[{"device": torch.device("cuda:0")}, {"device": torch.device("cuda:1")}])(x, y=y)
        ```

        Args:
            func: The function that should be wrapped.
            environments: The environments that the function should be run in, passed to the function as func(*args, **kwargs, **environment) for each environment, in the order specified.

        Returns:
            Wrapped function that can be called.
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for environment in self.environments:
                try:
                    return func(*args, **kwargs, **environment)
                except Exception as e:
                    logger.info(
                        f"Could not run the function {func} with environment"
                        f" {environment} (exception {e})"
                    )
                    continue
            logger.error(
                f"Could not run the function {func} in any specified environment"
            )
            return None

        return wrapper


def try_devices(
    func: Callable[P, R],
    devices: Collection[None | torch.device | str] = DEFAULT_DEVICES,
    *,
    current: bool = False,
) -> Callable[P, R]:
    """Tries to run a function on any of the provided `devices`, exiting on success.

    For each device provided, the tensor-valued arguments and keyword arguments to `func` are copied to the device before the function is run. This allows us to write device-agnostic code, since the function can be run on whichever device is available at runtime. This function can be used as a decorator.

    Example:

    ```
    import torch

    n = 10
    x = torch.zeros(n).to("cuda:1")
    y = torch.ones(n).to("cuda:0")

    # to try running the function on all devices

    @try_devices
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # to try running the function only on "cuda:1" and "cpu"

    @try_devices(devices=("cuda:1", "cpu"))
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # using the function without a decorator

    z = try_devices(add)(x, y=y)
    ```

    If you need more flexibility in how the function should be applied on different devices (for e.g., your function takes in numpy arrays and not tensors as inputs), consider using the function `try_environments`.

    Args:
        func: The function that should be wrapped.
        devices: GPUs/CPU that the function should be tried on, in the order specified. Defaults to all the GPUs available and then the CPU (i.e., ["cuda:0", ..., f"cuda:{torch.cuda.device_count()}", "cpu"])
        current: Whether to try running the function with all the tensors on their current devices, defaults to False.

    Returns:
        Wrapped function that can be called.
    """
    if not devices:  # handle case with empty Collection
        devices = DEFAULT_DEVICES
    else:
        devices = [
            None if device is None else torch.device(device) for device in devices
        ]

    if current:
        devices.insert(0, None)  # type:ignore  # devices is now a list

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        contains_tensor_arg = any(
            [isinstance(arg, torch.Tensor) for arg in args]
            + [isinstance(kwarg, torch.Tensor) for kwarg in kwargs.values()]
        )
        if not contains_tensor_arg:
            logger.warning(
                f"The function {func} does not have any tensor-valued arg/kwarg:"
                " `try_devices` is redundant"
            )

        for device in devices:
            try:
                args_device = [
                    arg.to(device) if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ]
                kwargs_device = {
                    key: kwarg.to(device) if isinstance(kwarg, torch.Tensor) else kwarg
                    for key, kwarg in kwargs.items()
                }

                return func(
                    *args_device,  # type:ignore  # args_device is guaranteed to have the same type as args
                    **kwargs_device,  # type:ignore  # kwargs_device is guaranteed to have the same type as kwargs
                )
            except Exception as e:
                logger.info(
                    f"Could not run the function {func} with device"
                    f" {device} (exception {e})"
                )
                try:
                    del args_device
                    del kwargs_device
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e2:
                    continue

                continue
        return None

    return wrapper
