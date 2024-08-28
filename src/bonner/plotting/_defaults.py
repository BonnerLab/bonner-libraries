from matplotlib.font_manager import get_font_names

from bonner.plotting._fonts import install_newcomputermodern

DEFAULT_FONTS = {
    "font.family": ["sans-serif", "serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

install_newcomputermodern()
if "NewComputerModernMath" in get_font_names():
    DEFAULT_FONTS |= {
        "font.serif": ["NewComputerModernMath"],
        "font.sans-serif": ["NewComputerModernSans10"],
    }

DEFAULT_SIZES = {
    "font.size": 10,
    "figure.figsize": (3, 3),
    "axes.labelsize": "medium",
    "figure.titlesize": "medium",
    "figure.labelsize": "medium",
    "legend.title_fontsize": "small",
    "legend.fontsize": "x-small",
    "xtick.major.size": 3.5,
    "xtick.major.width": 1,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1,
}

DEFAULT_FIGURE_OPTIONS = {
    "figure.autolayout": True,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "savefig.format": "svg",
    "savefig.transparent": True,
    "figure.facecolor": "None",
    "axes.facecolor": "None",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.edgecolor": "None",
    "legend.fancybox": False,
    "legend.frameon": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "patch.linewidth": 0,
}
DEFAULT_MATPLOTLIBRC = DEFAULT_FONTS | DEFAULT_SIZES | DEFAULT_FIGURE_OPTIONS

MM_PER_PT = 0.35277777777778
PT_PER_MM = 1 / MM_PER_PT
THICKNESS_IN_MM = 1.5
THICKNESS_IN_PTS = THICKNESS_IN_MM * PT_PER_MM

POSTER_SIZES = {
    "axes.linewidth": THICKNESS_IN_PTS,
    "xtick.major.width": THICKNESS_IN_PTS,
    "ytick.major.width": THICKNESS_IN_PTS,
    "axes.labelpad": 20,
    "axes.titlepad": 20,
    "xtick.major.pad": 16,
    "ytick.major.pad": 16,
    "font.size": 32,
    "axes.labelsize": 48,
    "axes.titlesize": 48,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 36,
    "legend.borderaxespad": 0,
}
