from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def save_figure(
    fig: Figure,
    *,
    filepath: Path,
    strip_metadata: bool = True,
    **kwargs: Any,
) -> None:
    filepath.parent.mkdir(exist_ok=True, parents=True)
    if strip_metadata:
        match filepath.suffix:
            case ".pdf":
                kwargs |= {
                    "metadata": {
                        "Creator": None,
                        "Producer": None,
                        "CreationDate": None,
                    },
                }
            case ".svg":
                kwargs |= {
                    "metadata": {
                        "Creator": None,
                        "Date": None,
                        "Format": None,
                        "Type": None,
                    },
                }
    fig.savefig(filepath, **kwargs)
