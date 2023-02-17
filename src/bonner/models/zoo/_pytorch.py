import typing
from typing import Any
import importlib
from collections.abc import Callable

import torch
from torchvision.models._api import Weights
from PIL import Image


def load(
    *, architecture: str, weights: str, **kwargs: Any
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    try:
        model_class = _load_architecture(architecture)
    except Exception:
        raise ValueError(f"model architecture {architecture} not found")

    preprocess = None
    match weights:
        case "untrained":
            seed = kwargs["seed"] if "seed" in kwargs.keys() else 0
            torch.manual_seed(seed)
            model = model_class()
        case _:
            try:
                weights = _load_weights(architecture)[weights]
                preprocess = weights.transforms()
            except Exception:
                raise ValueError(
                    f"weights {weights} not found for architecture {architecture}"
                )
            model = model_class(weights=weights)

    if preprocess is None:
        preprocess = _load_default_preprocess(architecture=architecture)
    return model, preprocess


def _load_weights(architecture: str) -> Weights:
    module = importlib.import_module("torchvision.models")
    weights = getattr(module, f"{architecture}_Weights")
    return weights


def _load_architecture(architecture: str) -> Callable[..., torch.nn.Module]:
    module = importlib.import_module("torchvision.models")
    architecture = getattr(module, architecture.lower())
    return typing.cast(Callable[..., torch.nn.Module], architecture)


def _load_default_preprocess(
    *, architecture: str
) -> Callable[[Image.Image], torch.Tensor]:
    return _load_weights(architecture)["DEFAULT"].transforms()
