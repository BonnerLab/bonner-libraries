from typing import Any

import seaborn as sns

DEFAULT_RC = {
    "font.family": ["serif"],
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "figure.figsize": (3, 3),
    "savefig.bbox": "tight",
    "savefig.format": "svg",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "legend.title_fontsize": "small",
    "legend.fontsize": "x-small",
    "lines.markeredgewidth": 0,
}


def set_plotting_defaults(
    context: str = "paper",
    style: str = "white",
    rc: dict[str, Any] = DEFAULT_RC,
) -> None:
    sns.set_theme(
        context=context,
        style=style,
        rc=rc,
    )
