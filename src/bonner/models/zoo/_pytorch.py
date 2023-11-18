from collections.abc import Callable

import torch
from PIL import Image
from torchvision.models import get_model, get_model_weights


def load(
    *,
    architecture: str,
    weights: str,
    seed: int | None = None,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    preprocess = None
    match weights:
        case "untrained":
            if seed is None:
                error = "seed cannot be None if case == 'untrained'"
                raise ValueError(error)
            torch.manual_seed(seed)
            model = get_model(name=architecture.lower(), weights=None)
            preprocess = get_model_weights(architecture)["DEFAULT"].transforms()
        case _:
            model = get_model(name=architecture.lower(), weights=weights)
            preprocess = get_model_weights(architecture)[weights].transforms()

    return model, preprocess
