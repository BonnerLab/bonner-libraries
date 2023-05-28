from typing import Any
from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from zenodo_get import zenodo_get

from bonner.files import unzip
from bonner.models.utilities import BONNER_MODELS_HOME

MODEL_ZOO_CACHE = BONNER_MODELS_HOME / "models" / "model_zoo"

# tiny-imagenet_resnet18_kaiming_uniform_subset
DOI = "10.5281/zenodo.7023278"
ZIP_FILENAME = "tiny-imagenet_resnet18_subset.zip"
UNZIP_DIR = "tiny-imagenet_resnet18_kaiming_uniform_subset"


def load_model_zoo_checkpoint(*, seed: int) -> dict[str, Any]:
    zenodo_get(f"-d {DOI} -o {MODEL_ZOO_CACHE}".split())
    unzip(
        filepath=MODEL_ZOO_CACHE / ZIP_FILENAME,
        extract_dir=MODEL_ZOO_CACHE,
        remove_zip=False,
    )
    root = MODEL_ZOO_CACHE / UNZIP_DIR
    filepath = [path.relative_to(root) for path in root.rglob(f"*seed={seed}_*")][0]
    return torch.load(
        filepath / "checkpoint_000060" / "checkpoints",
        map_location=torch.device("cpu")
    )


def load(
    *, seed: int,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    assert seed >=1 and seed <= 106, "seed must be in [1, 106]"
    checkpoint = load_model_zoo_checkpoint(seed)

    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 200)
    model.load_state_dict(checkpoint)

    preprocess = torchvision.models.get_model_weights("resnet18")[
        "DEFAULT"
    ].transforms()
    return model, preprocess
