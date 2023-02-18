from collections.abc import Callable
from pathlib import Path

from PIL import Image
import torch
import torchvision

from bonner.files import download_from_url
from bonner.models.utilities import BONNER_MODELS_HOME

URLS = {
    "robust-epsilon=0": "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D",
    "robust-epsilon=3": "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps3.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D",
}

CUSTOM_CACHE = BONNER_MODELS_HOME / "models" / "custom"


def download_checkpoint(weights: str) -> Path:
    return download_from_url(
        url=URLS[weights],
        filepath=CUSTOM_CACHE / "downloads" / f"VGG16_bn.robust.{weights}.ckpt",
        force=False,
    )


def load(weights: str) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    checkpoint = torch.load(download_checkpoint(weights))
    model = torchvision.models.vgg16_bn()
    model.load_state_dict(checkpoint["model_state_dict"])
    return (
        model,
        torchvision.models.get_model_weights("vgg16_bn")["DEFAULT"].transforms(),
    )
