from collections.abc import Callable
import logging

import torch
import timm

logging.getLogger("timm").setLevel(logging.WARNING)

from torchvision.models import get_model_weights
from PIL import Image


def load(
    *, architecture: str, pretrained: bool = True, seed: int = None,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    if not pretrained:
        assert seed is not None
        torch.manual_seed(seed)
    model = timm.create_model(
        architecture, 
        pretrained=pretrained,
    )
    # default preprocess
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    
    
    # TODO: test old preprocess
    # preprocess = get_model_weights("resnet18")["DEFAULT"].transforms()
    return model, preprocess
