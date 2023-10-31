import os
from pathlib import Path

import torch
from bonner.models.datapipes import create_image_datapipe
from bonner.models.features import extract_features
from bonner.models.hooks import GlobalMaxpool
from PIL import Image
from torchvision.datasets import DTD
from torchvision.models import AlexNet_Weights, alexnet

weights = AlexNet_Weights.IMAGENET1K_V1


def preprocess(image: Image) -> torch.Tensor:
    return weights.transforms()(image)


dataset = DTD(
    os.getenv("BONNER_DATASETS_HOME", str(Path.home() / ".cache" / "bonner-datasets")),
    download=True,
)
stimuli = list((Path(dataset.root) / "dtd" / "dtd" / "images").rglob("*.jpg"))

datapipe = create_image_datapipe(
    image_paths=stimuli,
    image_ids=[stimulus.name for stimulus in stimuli],
    preprocess_fn=preprocess,
    batch_size=64,
)

extractor = extract_features(
    model=alexnet(weights=weights),
    model_identifier="AlexNet-IMAGENET1K_V1",
    nodes=[
        "features.1",
        "features.4",
    ],
    hooks={
        "features.1": GlobalMaxpool(),
    },
    datapipe=datapipe,
    datapipe_identifier="dtd",
)
