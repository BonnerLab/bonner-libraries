from pathlib import Path
from collections.abc import Callable

from PIL import Image
import numpy as np
import torch
import torchvision
import xarray as xr
from bonner.files import download_from_url
from bonner.caching import cache
from bonner.models.utilities import BONNER_MODELS_HOME

# https://github.com/GKalliatakis/Keras-VGG16-places365/issues/5
# https://github.com/antorsae/landmark-recognition-challenge/blob/master/extra/vgg16_places365.py
# https://github.com/microsoft/robust-models-transfer

URLS = {
    "Places365": "https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5",
    "Hybrid1365": "https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels.h5",
}

CUSTOM_CACHE = BONNER_MODELS_HOME / "models" / "custom"

MAPPING = {
    "features.0": "block1_conv1",
    "features.2": "block1_conv2",
    "features.5": "block2_conv1",
    "features.7": "block2_conv2",
    "features.10": "block3_conv1",
    "features.12": "block3_conv2",
    "features.14": "block3_conv3",
    "features.17": "block4_conv1",
    "features.19": "block4_conv2",
    "features.21": "block4_conv3",
    "features.24": "block5_conv1",
    "features.26": "block5_conv2",
    "features.28": "block5_conv3",
    "classifier.0": "fc1",
    "classifier.3": "fc2",
    "classifier.6": "predictions",
}


def download_model(weights: str) -> Path:
    return download_from_url(
        url=URLS[weights],
        filepath=CUSTOM_CACHE / "downloads" / f"VGG16.{weights}.h5",
        force=False,
    )


def load(weights: str) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    state_dict = create_state_dict(weights)
    model = torchvision.models.vgg16(num_classes=365)
    model.load_state_dict(state_dict)
    return model, preprocess


@cache(
    "state_dicts/architecture=VGG16.weights={weights}.pkl",
    path=CUSTOM_CACHE,
)
def create_state_dict(weights: str) -> dict[str, torch.Tensor]:
    filepath = download_model(weights)
    state_dict = {}
    for torch_layer, keras_layer in MAPPING.items():
        group = f"/{keras_layer}/{keras_layer}"
        dataset = xr.open_dataset(filepath, group=group)

        kernel = torch.from_numpy(dataset["kernel:0"].values)
        if kernel.ndim == 4:
            state_dict[f"{torch_layer}.weight"] = kernel.permute(3, 2, 0, 1)
        elif kernel.ndim == 2:
            state_dict[f"{torch_layer}.weight"] = kernel.permute(1, 0)

        bias = torch.from_numpy(dataset["bias:0"].values)
        state_dict[f"{torch_layer}.bias"] = bias
    return state_dict


def preprocess(image: Image.Image) -> torch.Tensor:
    resizer = torchvision.transforms.Resize(size=(224, 224), antialias=True)
    image_ = torch.from_numpy(np.array(resizer(image)))
    image_ = (image_ - image_.min()) * 255 / (image_.max() - image_.min())
    image_ -= torch.Tensor([123.68, 116.779, 103.939])
    image_ = image_.flip(dims=[-1]).permute((2, 0, 1))
    return image_
