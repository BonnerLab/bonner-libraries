from typing import Any
from collections.abc import Callable

import torch
import torchvision
from PIL import Image

from bonner.files import download_from_url
from bonner.models.utilities import BONNER_MODELS_HOME

VISSL_CACHE = BONNER_MODELS_HOME / "models" / "vissl"
URLS = {
    "Jigsaw-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch",
    "Colorization-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in1k_goyal19.torch",
    "RotNet-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
    "ClusterFit-16K-RotNet-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch",
    "NPID++-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch",
    "PIRL-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
    "SimCLR-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
    "SwAV-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch",
    "MoCoV2-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
    "BarlowTwins-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
    # 
    "DeepClusterV2-ImageNet1K": "https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar",
    # 
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
                    "Jigsaw-ImageNet1K"
                    | "Colorization-ImageNet1K"
                    | "RotNet-ImageNet1K"
                    | "ClusterFit-16K-RotNet-ImageNet1K"
                    | "NPID++-ImageNet1K"
                    | "PIRL-ImageNet1K"
                    | "SimCLR-ImageNet1K"
                    | "SwAV-ImageNet1K"
                    | "MoCoV2-ImageNet1K"
                    | "BarlowTwins-ImageNet1K"
                ):
                    checkpoint = checkpoint["classy_state_dict"]["base_model"]["model"][
                        "trunk"
                    ]
                case "DeepClusterV2-ImageNet1K":
                    checkpoint = checkpoint
                case "Instagram-ImageNet" | "YFCC100M-ImageNet" | "Places205-Caffe2":
                    checkpoint = checkpoint["model_state_dict"]
                case _:
                    raise ValueError(
                        f"architecture {architecture} has no weights '{weights}'"
                    )
        case _:
            raise ValueError(f"architecture {architecture} not defined")

    new_state_dict = {}
    match weights:
        case "DeepClusterV2-ImageNet1K":
            for key in checkpoint.keys():
                new_state_dict[key.replace("module.", "")] = checkpoint[key]
        case _:
            for key in checkpoint.keys():
                new_state_dict[key.replace("_feature_blocks.", "")] = checkpoint[key]
    model = torchvision.models.resnet50()
    model.load_state_dict(new_state_dict, strict=False)

    preprocess = torchvision.models.get_model_weights(architecture)[
        "DEFAULT"
    ].transforms()
    return model, preprocess
