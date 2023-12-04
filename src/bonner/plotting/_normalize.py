import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize


def create_centered_diverging_cmap(
    data: np.ndarray,
    /,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float = 0,
    n_colors: int = 256,
) -> ListedColormap:
    """Create a cmap where bars diverge from a specified center.

    Args:
    ----
        data: Data for which the cmap is created
        cmap: Colormap to use, preferably a diverging colormap (e.g. RdBu_r)
        vmin: Custom minimum value of the colorbar. Defaults to `data.min()`.
        vmax: Custom maximum value of the colorbar. Defaults to `data.max()`.
        center: Value where the colorbar should diverge from. Defaults to 0.
        n_colors: Number of colors to use for the colormap. Defaults to 256.

    Returns:
    -------
        Customized diverging colormap with appropriate center and limits.

    Example:
    -------
    import numpy as np
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(seed=0)
    x = 50 * rng.standard_normal((200, 200))

    cmap = create_centered_diverging_cmap(
        x,
        center=0,
        vmin=-50,
        vmax=100,
    )

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap=cmap)
    """
    # TODO validate that vmin and vmax work properly
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    vrange = max(vmax - center, center - vmin)
    normalize = Normalize(center - vrange, center + vrange)
    cmin, cmax = normalize([vmin, vmax])
    cc = np.linspace(cmin, cmax, n_colors)
    return ListedColormap(sns.color_palette(cmap, as_cmap=True)(cc))
