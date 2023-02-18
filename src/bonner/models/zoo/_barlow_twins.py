from collections.abc import Callable

import torch
from PIL import Image
from torchvision.models import get_model_weights


def load() -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
    preprocess = get_model_weights("ResNet50")["DEFAULT"].transforms()
    return model, preprocess
