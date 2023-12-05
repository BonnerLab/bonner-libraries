from PIL import Image


def fill_transparent_background(
    input_: Image.Image,
    /,
    *,
    color: str = "WHITE",
) -> Image.Image:
    output = Image.new("RGBA", input_.size, color)
    output.paste(input_, mask=input_)
    return output
