from typing import Any
import functools
from collections.abc import Callable

import torch
import torchvision
from PIL import Image

from bonner.files import download_from_url
from bonner.models.utilities import BONNER_MODELS_HOME
from bonner.models.zoo._pytorch import _preprocess

VISSL_CACHE = BONNER_MODELS_HOME / "models" / "vissl"
URLS = {
    "ClusterFit-16K-RotNet-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch",
    "NPID++-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch",
    "PIRL-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
    "SimCLR-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
    "SwAV-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch",
    "Instagram-ImageNet": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch",
    "YFCC100M-ImageNet": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_semi_sup_08389792.torch",
    "Places205-Caffe2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_supervised_places205_caffe2.torch",
}


def load_vissl_checkpoint(*, architecture: str, weights: str) -> dict[str, Any]:
    filepath = download_from_url(
        URLS[weights],
        filepath=VISSL_CACHE / f"{architecture}-{weights}.pth",
        force=False,
    )
    return torch.load(filepath, map_location=torch.device("cpu"))


def load(
    *, architecture: str, weights: str
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    checkpoint = load_vissl_checkpoint(architecture=architecture, weights=weights)

    match architecture:
        case "ResNet50":
            match weights:
                case (
                    "ClusterFit-16K-RotNet-ImageNet1K"
                    | "NPID++-ImageNet1K"
                    | "PIRL-ImageNet1K"
                    | "SimCLR-ImageNet1K"
                    | "SwAV-ImageNet1K"
                ):
                    checkpoint = checkpoint["classy_state_dict"]["base_model"]["model"][
                        "trunk"
                    ]
                case "Instagram-ImageNet" | "YFCC100M-ImageNet" | "Places205-Caffe2":
                    checkpoint = checkpoint["model_state_dict"]
                case _:
                    raise ValueError(
                        f"architecture {architecture} has no weights '{weights}'"
                    )
        case "ResNet18":
            match weights:
                case _:
                    raise ValueError(
                        f"architecture {architecture} has no weights '{weights}'"
                    )
        case _:
            raise ValueError(f"architecture {architecture} not defined")

    new_state_dict = {}
    for key in checkpoint.keys():
        new_state_dict[key.replace("_feature_blocks.", "")] = checkpoint[key]
    model = torchvision.models.resnet50()
    model.load_state_dict(new_state_dict, strict=False)

    preprocess = functools.partial(_preprocess, architecture=architecture)
    return model, preprocess
