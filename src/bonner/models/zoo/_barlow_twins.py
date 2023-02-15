import functools
from collections.abc import Callable

import torch
from PIL import Image

from bonner.models.zoo._pytorch import _preprocess


def load() -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
    preprocess = functools.partial(_preprocess, architecture="ResNet50")
    return model, preprocess
