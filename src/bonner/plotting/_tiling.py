from collections.abc import Sequence
from pathlib import Path

from PIL import Image


def concatenate_images(
    image_paths: Sequence[Path],
    *,
    direction: str,
    resize: bool = True,
) -> Image.Image:
    images = [Image.open(image_path) for image_path in image_paths]
    sizes = [image.size for image in images]

    ws = [size[0] for size in sizes]
    hs = [size[1] for size in sizes]

    w_max = max(ws)
    h_max = max(hs)

    match direction:
        case "horizontal":
            w = sum([int(w * h_max / h) for w, h in zip(ws, hs)]) if resize else sum(ws)
            shape = (w, h_max)
        case "vertical":
            h = sum([int(h * w_max / w) for w, h in zip(ws, hs)]) if resize else sum(hs)
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
