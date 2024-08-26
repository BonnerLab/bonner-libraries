import functools
from collections.abc import Sequence

from PIL import Image


def concatenate_multiple_images(
    images: Sequence[Image.Image],
    *,
    direction: str,
    resize: bool = True,
) -> Image.Image:
    sizes = [image.size for image in images]

    ws = [size[0] for size in sizes]
    hs = [size[1] for size in sizes]

    w_max = max(ws)
    h_max = max(hs)

    match direction:
        case "horizontal":
            w = (
                sum([int(w * h_max / h) for w, h in zip(ws, hs, strict=True)])
                if resize
                else sum(ws)
            )
            shape = (w, h_max)
        case "vertical":
            h = (
                sum([int(h * w_max / w) for w, h in zip(ws, hs, strict=True)])
                if resize
                else sum(hs)
            )
            shape = (w_max, h)
        case _:
            raise ValueError

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


def rescale_image(
    image: Image.Image,
    /,
    *,
    size: tuple[int, int],
    direction: str,
) -> Image.Image:
    w_reference, h_reference = size
    w, h = image.size
    match direction:
        case "height":
            return image.resize(
                (int(w * h_reference / h), h_reference),
            )
        case "width":
            return image.resize(
                (w_reference, int(h * w_reference / w)),
            )
        case _:
            raise ValueError


def concatenate_images(
    first: Image.Image,
    second: Image.Image,
    /,
    *,
    direction: str,
    overlap: float = 0,
    reverse_zorder: bool = False,
    color: str | None = "white",
) -> Image.Image:
    match direction:
        case "horizontal":
            size = (
                first.width + int((1 - overlap) * second.width),
                first.height,
            )
            location = (size[0] - second.width, 0)

        case "vertical":
            size = (
                first.width,
                first.height + int((1 - overlap) * second.height),
            )
            location = (0, size[1] - second.height)
        case _:
            raise ValueError

    if color is not None:
        concatenated_image = Image.new(mode="RGBA", size=size, color=color)
    else:
        concatenated_image = Image.new(mode="RGBA", size=size)

    paste_second_image = functools.partial(
        concatenated_image.alpha_composite,
        second,
        location,
    )

    if reverse_zorder:
        concatenated_image.alpha_composite(first, (0, 0))
        paste_second_image()
    else:
        paste_second_image()
        concatenated_image.alpha_composite(first, (0, 0))

    return concatenated_image


def crop_fraction(
    image: Image.Image,
    *,
    left: float = 0,
    right: float = 0,
    top: float = 0,
    bottom: float = 0,
) -> Image.Image:
    width, height = image.size
    return image.crop(
        (
            int(left * width),
            int(top * height),
            int(width - right * width),
            int(height - bottom * height),
        ),
    )
