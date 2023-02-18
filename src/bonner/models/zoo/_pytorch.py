from collections.abc import Callable

import torch
from torchvision.models import get_model, get_model_weights
from PIL import Image


def load(
    *, architecture: str, weights: str, seed: int = 0
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    preprocess = None
    match weights:
        case "untrained":
            torch.manual_seed(seed)
            model = get_model(name=architecture.lower(), weights=None)
            preprocess = get_model_weights(architecture)["DEFAULT"].transforms()
        case _:
            model = get_model(name=architecture.lower(), weights=weights)
            preprocess = get_model_weights(architecture)[weights].transforms()

    return model, preprocess
