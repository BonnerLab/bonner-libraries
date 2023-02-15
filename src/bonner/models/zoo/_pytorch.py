import typing
from typing import Any
import importlib
from collections.abc import Callable
import functools

import torch
from PIL import Image


def load(
    *, architecture: str, weights: str, **kwargs: Any
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    match architecture:
        case "AlexNet" | "ResNet18" | "ResNet50" | "VGG16" | "VGG19":
            match weights:
                case "untrained":
                    seed = kwargs["seed"] if "seed" in kwargs.keys() else 0
                    torch.manual_seed(seed)
                    model = _load_architecture(architecture)()
                case "ImageNet":
                    weights = _load_weights(architecture)["IMAGENET1K_V1"]
                    model = _load_architecture(architecture)(weights=weights)
                case _:
                    raise ValueError("model not defined")

            preprocess = functools.partial(_preprocess, architecture=architecture)
            return model, preprocess
        case "ViT_b_16" | "ViT_b_32" | "ViT_l_16" | "ViT_l_32" | "ViT_h_14":
            raise NotImplementedError()
        case _:
            raise ValueError("architecture not known")


def _load_weights(architecture: str):
    module = importlib.import_module("torchvision.models")
    weights = getattr(module, f"{architecture}_Weights")
    return weights


def _load_architecture(architecture: str) -> Callable[..., torch.nn.Module]:
    module = importlib.import_module("torchvision.models")
    architecture = getattr(module, architecture.lower())
    return typing.cast(Callable[..., torch.nn.Module], architecture)


def _preprocess(image: Image.Image, *, architecture: str) -> torch.Tensor:
    image = image.convert("RGB")
    weights = _load_weights(architecture)["IMAGENET1K_V1"]
    return weights.transforms()(image)
