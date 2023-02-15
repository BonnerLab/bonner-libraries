from typing import Any
from collections.abc import Callable
import functools

import torch
from PIL import Image
from robustbench.utils import load_model

from bonner.models.utilities import BONNER_MODELS_CACHE
from bonner.models.zoo._pytorch import _preprocess


def load_robustbench_model(**kwargs: Any) -> torch.nn.Module:
    robustbench_dir = BONNER_MODELS_CACHE / "models" / "robustbench"
    robustbench_dir.mkdir(exist_ok=True, parents=True)
    return load_model(model_dir=robustbench_dir, **kwargs).model


def load(
    *, architecture: str, weights: str
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    match architecture:
        case "ResNet18":
            match weights:
                case "Salman2020":
                    model = load_robustbench_model(
                        model_name="Salman2020Do_R18",
                        dataset="imagenet",
                        threat_model="Linf",
                    )
                case _:
                    raise ValueError()
        case "ResNet50":
            match weights:
                case "Wong2020":
                    model = load_robustbench_model(
                        model_name="Wong2020Fast",
                        dataset="imagenet",
                        threat_model="Linf",
                    )
                case _:
                    raise ValueError()
        case _:
            raise ValueError()

    preprocess = functools.partial(_preprocess, architecture=architecture)
    return model, preprocess
