from bonner.plotting._fonts import install_newcomputermodern
from matplotlib.font_manager import get_font_names

install_newcomputermodern()
if "NewComputerModernMath" not in get_font_names():
    kwargs = {
        "font.serif": ["NewComputerModernMath"],
        "font.sans-serif": ["NewComputerModernSans10"],
    }
else:
    kwargs = {}

DEFAULT_MATPLOTLIBRC = kwargs | {
    "font.family": ["sans-serif", "serif"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.figsize": (3, 3),
    "figure.autolayout": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "savefig.format": "svg",
    "savefig.transparent": True,
    "figure.facecolor": "None",
    "axes.facecolor": "None",
    "axes.labelsize": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "figure.titlesize": "medium",
    "figure.labelsize": "medium",
    "legend.title_fontsize": "small",
    "legend.fontsize": "x-small",
    "legend.fancybox": False,
    "legend.frameon": False,
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
