from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(
    mappable: ScalarMappable,
    *,
    ax: Axes,
    label: str = "",
    location: str = "right",
    rotation: float = -90,
    size: str = "2%",
    pad: float = 0.2,
    outline: bool = False,
    ticks: bool = False,
) -> tuple[Colorbar, Axes]:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    fig = ax.get_figure()
    cb = fig.colorbar(mappable=mappable, cax=cax, location=location)
    cb.set_label(label, rotation=rotation)
    cb.outline.set_visible(outline)
    if not ticks:
        cax.tick_params(length=0)
    return cb, cax
