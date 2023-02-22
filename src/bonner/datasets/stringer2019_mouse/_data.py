from pathlib import Path

import pandas as pd
from scipy.io import loadmat
import xarray as xr

from PIL import Image

from bonner.files._figshare import get_url_dict
from bonner.files import download_from_url
from bonner.datasets.stringer2019_mouse._utilities import CACHE_PATH

FIGSHARE_ARTICLE_ID = 6845348


def _download_dataset(force: bool = False) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID)
    urls_data = {key: url for key, url in urls.items() if "natimg2800_M" in key}
    for filename, url in urls_data.items():
        download_from_url(url, filepath=CACHE_PATH / filename, force=force)
    filename = "images_natimg2800_all.mat"
    download_from_url(urls[filename], filepath=CACHE_PATH / filename, force=force)


def create_data_assembly(*, mouse: str, date: str) -> xr.Dataset:
    _download_dataset()
    raw = loadmat(CACHE_PATH / f"natimg2800_{mouse}_{date}.mat", simplify_cells=True)
    return xr.Dataset(
        data_vars={
            "stimulus-related activity": (
                ("presentation", "neuroid"),
                raw["stim"]["resp"],
            ),
            "spontaneous activity": (("time", "neuroid"), raw["stim"]["spont"]),
        },
        coords={
            "stimulus_id": (
                "presentation",
                [
                    "blank" if i_image == 2800 else f"image{i_image:04}"
                    for i_image in raw["stim"]["istim"] - 1
                ],
            ),
            "x": ("neuroid", raw["med"][:, 0]),
            "y": ("neuroid", raw["med"][:, 1]),
            "z": ("neuroid", raw["med"][:, 2]),
            "noise_level": (
                "neuroid",
                pd.DataFrame(raw["stat"])["noiseLevel"].values,
            ),
        },
    )


def _save_images() -> tuple[Path, list[Path]]:
    images = loadmat(CACHE_PATH / "images_natimg2800_all.mat", simplify_cells=True)[
        "imgs"
    ]
    path = CACHE_PATH / "images"
    path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i_image in range(images.shape[-1]):
        path_ = path / f"image{i_image:04}.png"
        paths.append(path_)
        if not path_.exists():
            Image.fromarray(images[:, :, i_image]).convert("RGB").save(path_)
    return path, paths


def create_stimulus_set() -> tuple[Path, pd.DataFrame]:
    _download_dataset()
    path, paths = _save_images()
    return path, pd.DataFrame(
        {
            "stimulus_id": [path.stem for path in paths],
            "filename": paths,
        }
    ).set_index("stimulus_id")
