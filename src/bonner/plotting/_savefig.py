from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(fig: Figure, *, filepath: Path, **kwargs: Any) -> None:
    filepath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(filepath, **kwargs)
