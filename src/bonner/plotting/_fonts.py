import os
import platform
import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
from bonner.files import download_from_url, unzip
from matplotlib.font_manager import get_font, get_font_names


def install_newcomputermodern() -> None:
    """Install the NewComputerModernMath font and register it with matplotlib.

    A no-op when the font is already available, and Linux-only — elsewhere it warns and returns.

    Installing clears matplotlib's cache directory, so the font cache is rebuilt on next use, and
    writes a fontconfig file under the user's config directory.

    Never raises. The font is cosmetic, and failing to fetch it must not stop a non-plotting
    workload from importing the package, so every failure becomes a warning.
    """
    if platform.system() != "Linux":
        warnings.warn(
            "NewComputerModernMath font can only be installed automatically on Linux",
        )
        return

    font_name = "NewComputerModernMath"
    if font_name in get_font_names():
        return

    try:
        if Path(mpl.get_cachedir()).exists():
            shutil.rmtree(mpl.get_cachedir())

        data_home = Path(
            os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share")),
        )
        data_path = data_home / "fonts"
        data_path.mkdir(exist_ok=True, parents=True)

        filepath = download_from_url(
            "https://mirrors.ctan.org/fonts/newcomputermodern.zip",
            filepath=data_path / "newcomputermodern.zip",
        )
        filepath = unzip(filepath, extract_dir=data_path)

        config_home = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        config_path = config_home / "fontconfig" / "fonts.conf"
        config_path.parent.mkdir(exist_ok=True, parents=True)

        with config_path.open("w") as f:
            f.write('<dir prefix="xdg">fonts</dir>')

        get_font(font_name)
    except Exception as exc:  # noqa: BLE001 — font install is cosmetic, never fatal
        warnings.warn(f"could not auto-install {font_name} font: {exc}")
