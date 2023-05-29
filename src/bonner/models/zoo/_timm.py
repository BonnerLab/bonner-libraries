from collections.abc import Callable

import torch
import timm
from torchvision.models import get_model_weights
from PIL import Image


def load(
    *, architecture: str
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    model = timm.create_model(
        architecture, 
        pretrained=True
    )
    # default preprocess
    preprocess = get_model_weights("resnet18")["DEFAULT"].transforms()
    return model, preprocess
