from bonner.plotting._fonts import install_newcomputermodern

DEFAULT_MATPLOTLIBRC = {
    "font.family": ["serif"],
    "font.serif": ["NewComputerModernMath"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "figure.figsize": (3, 3),
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.format": "svg",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "figure.titlesize": "medium",
    "figure.labelsize": "medium",
    "legend.title_fontsize": "small",
    "legend.fontsize": "x-small",
    "lines.markeredgewidth": 0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "patch.linewidth": 0,
}

install_newcomputermodern()
