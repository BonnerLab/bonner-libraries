from typing import Any
from collections.abc import Callable

import torch
import torchvision
from PIL import Image

from bonner.files import download_from_url
from bonner.models.utilities import BONNER_MODELS_HOME

WEIGHTS = [
    "ref01",
    "ref06_diet_imagenet",
    "ref07_diet_openimagesv6",
    "ref08_diet_places2",
    "ref09_diet_vggface2",
    "ref10_diet_FacesPlacesObjects1281167",
    "ref11_diet_FacesPlacesObjects1281167x3"
]


def load(
    *, architecture: str, weights: str
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    assert architecture == "alexnetgn"
    assert weights in WEIGHTS
    return torch.hub.load("harvard-visionlab/open_ipcl", f"alexnetgn_ipcl_{weights}")