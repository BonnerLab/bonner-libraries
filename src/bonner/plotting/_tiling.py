from typing import Any
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from matplotlib.colors import Colormap
import seaborn as sns


def set_style(
    context: str = "paper",
    style: str = "white",
    palette: list[tuple[float, float, float]]
    | Colormap = sns.color_palette("colorblind"),
    rc: dict[str, Any] = {},
) -> None:
    default_rc = {
        "font.family": ["sans-serif"],
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "Verdana",
            "Computer Modern Sans Serif",
        ],
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 28,
        "font.size": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "savefig.bbox": "tight",
    }

    sns.set_context(context)
    sns.set_style(
        style=style,
        rc=default_rc | rc,
    )
    sns.set_palette(palette)



def concatenate_images(
    image_paths: Sequence[Path], *, direction: str, resize: bool = True
) -> Image.Image:
    images = [Image.open(image_path) for image_path in image_paths]
    sizes = [image.size for image in images]

    ws = [size[0] for size in sizes]
    hs = [size[1] for size in sizes]

    w_max = max(ws)
    h_max = max(hs)

    match direction:
        case "horizontal":
            if resize:
                w = sum([int(w * h_max / h) for w, h in zip(ws, hs)])
            else:
                w = sum(ws)
            shape = (w, h_max)
        case "vertical":
            if resize:
                h = sum([int(h * w_max / w) for w, h in zip(ws, hs)])
            else:
                h = sum(hs)
            shape = (w_max, h)

    concatenated_image = Image.new("RGB", shape)

    current = 0
    for image in images:
        w, h = image.size

        match direction:
            case "horizontal":
                if resize:
                    image = image.resize((int(w * h_max / h), h_max))
                    location = (current, 0)
                else:
                    location = (current, (h_max - h) // 2)
                current += image.size[0]
            case "vertical":
                if resize:
                    image = image.resize((w_max, int(h * w_max / w)))
                    location = (0, current)
                else:
                    location = ((w_max - w) // 2, current)
                current += image.size[1]
        concatenated_image.paste(image, location)
    return concatenated_image
