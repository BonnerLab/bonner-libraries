import os
import platform
import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
from matplotlib.font_manager import get_font, get_font_names

from bonner.files import download_from_url, unzip


def install_newcomputermodern() -> None:
    if platform.system() != "Linux":
        warnings.warn(
            "NewComputerModernMath font can only be installed automatically on Linux",
            stacklevel=2,
        )
        return

    font_name = "NewComputerModernMath"
    if font_name not in get_font_names():
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
