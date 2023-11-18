from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision
import xarray as xr
from bonner.files import download_from_url
from bonner.models.utilities import BONNER_MODELS_HOME
from PIL import Image

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


def download_weights(weights: str) -> Path:
    return download_from_url(
        url=URLS[weights],
        filepath=CUSTOM_CACHE / "downloads" / f"VGG16.{weights}.h5",
        force=False,
    )


def load(weights: str) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    state_dict = create_state_dict(weights)
    match weights:
        case "Places365":
            n_classes = 365
        case "Hybrid1365":
            n_classes = 1365
    model = torchvision.models.vgg16(num_classes=n_classes)
    model.load_state_dict(state_dict)
    return model, preprocess


def create_state_dict(weights: str) -> dict[str, torch.Tensor]:
    filepath = download_weights(weights)
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
    image_ = torch.from_numpy(np.asarray(resizer(image)).astype(np.float32))
    image_ = (image_ - image_.min()) / (image_.max() - image_.min())
    image_ -= torch.Tensor([0.485, 0.456, 0.406])

    # (H, W, C) -> (C, H, W)
    image_ = image_.permute((2, 0, 1))
    # RGB -> BGR
    image_ = image_.flip(0)

    return image_


if __name__ == "__main__":
    TEST_IMAGE_URL = "http://places2.csail.mit.edu/imgs/demo/6.jpg"
    CLASSES_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

    image_filepath = download_from_url(TEST_IMAGE_URL)
    classes_filepath = download_from_url(CLASSES_URL)

    classes = []
    with classes_filepath.open() as class_file:
        classes = [line.strip().split(" ")[0][3:] for line in class_file]
    classes = tuple(classes)

    model, preprocess_ = load("Places365")
    model.eval()
    x = preprocess_(Image.open(image_filepath)).unsqueeze(0)
    predictions = model(x)[0]

    n = 10
    top_predictions = torch.argsort(predictions, descending=True, stable=True)[:n]
    for i in range(n):
        print(classes[top_predictions[i]])
