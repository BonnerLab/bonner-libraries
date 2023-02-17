from typing import Any
from collections.abc import Callable

import torch
from PIL import Image
from robustbench.utils import load_model

from bonner.models.utilities import BONNER_MODELS_HOME
from bonner.models.zoo._pytorch import _load_default_preprocess


def load_robustbench_model(**kwargs: Any) -> torch.nn.Module:
    robustbench_dir = BONNER_MODELS_HOME / "models" / "robustbench"
    robustbench_dir.mkdir(exist_ok=True, parents=True)
    return load_model(model_dir=robustbench_dir, **kwargs).model


def load(
    *, architecture: str, weights: str
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    match (architecture, weights):
        case ("ResNet18", "Salman2020"):
            model = load_robustbench_model(
                model_name="Salman2020Do_R18",
                dataset="imagenet",
                threat_model="Linf",
            )
        case ("ResNet50", "Wong2020"):
            model = load_robustbench_model(
                model_name="Wong2020Fast",
                dataset="imagenet",
                threat_model="Linf",
            )
        case _:
            raise ValueError(
                f"architecture {architecture} with weights {weights} not found"
            )

    preprocess = _load_default_preprocess(architecture=architecture)
    return model, preprocess
