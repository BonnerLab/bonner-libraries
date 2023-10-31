from collections.abc import Callable
from typing import Any

import torch
import torchvision
from bonner.files import unzip
from bonner.models.utilities import BONNER_MODELS_HOME
from PIL import Image
from torch import nn
from zenodo_get import zenodo_get

MODEL_ZOO_CACHE = BONNER_MODELS_HOME / "models" / "model_zoo"

# tiny-imagenet_resnet18_kaiming_uniform_subset
DOI = "10.5281/zenodo.7023278"
ZIP_FILENAME = "tiny-imagenet_resnet18_subset.zip"
UNZIP_DIR = "tiny-imagenet_resnet18_kaiming_uniform_subset"


def load_model_zoo_checkpoint(seed: int) -> dict[str, Any]:
    if not (MODEL_ZOO_CACHE / UNZIP_DIR).exists():
        zenodo_get(f"-d {DOI} -o {MODEL_ZOO_CACHE}".split())
        unzip(
            filepath=MODEL_ZOO_CACHE / ZIP_FILENAME,
            extract_dir=MODEL_ZOO_CACHE,
            remove_zip=False,
        )
    root = MODEL_ZOO_CACHE / UNZIP_DIR
    filepath = [path.relative_to(root) for path in root.rglob(f"*seed={seed}_*")][0]
    return torch.load(
        MODEL_ZOO_CACHE / UNZIP_DIR / filepath / "checkpoint_000060" / "checkpoints",
        map_location=torch.device("cpu"),
    )


def load(
    *,
    seed: int,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    assert seed >= 1 and seed <= 106, "seed must be in [1, 106]"
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

    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(64),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )
    return model, preprocess
