from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from scipy.io import loadmat

from bonner.datasets.stringer2019_mouse._utilities import CACHE_PATH
from bonner.files import download_from_url
from bonner.files._figshare import get_url_dict

FIGSHARE_ARTICLE_ID = 6845348


def _download_dataset(*, force: bool = False) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID)
    urls_data = {key: url for key, url in urls.items() if "natimg2800_M" in key}
    for filename, url in urls_data.items():
        download_from_url(url, filepath=CACHE_PATH / filename, force=force)
    filename = "images_natimg2800_all.mat"
    download_from_url(urls[filename], filepath=CACHE_PATH / filename, force=force)


def create_data_assembly(*, mouse: str, date: str) -> xr.Dataset:
    _download_dataset()
    raw = loadmat(CACHE_PATH / f"natimg2800_{mouse}_{date}.mat", simplify_cells=True)
    assembly = xr.Dataset(
        data_vars={
            "stimulus-related activity": (
                ("presentation", "neuroid"),
                raw["stim"]["resp"],
            ),
            "spontaneous activity": (("time", "neuroid"), raw["stim"]["spont"]),
        },
        coords={
            "stimulus": (
                "presentation",
                (raw["stim"]["istim"] - 1).astype(np.uint16),
            ),
            "x": ("neuroid", raw["med"][:, 0]),
            "y": ("neuroid", raw["med"][:, 1]),
            "z": ("neuroid", raw["med"][:, 2]),
            "noise_level": (
                "neuroid",
                pd.DataFrame(raw["stat"])["noiseLevel"].to_numpy(),
            ),
        },
        attrs={"mouse": mouse, "date": date},
    )
    reps: dict[str, int] = {}
    repetitions: list[int] = []
    for stimulus in assembly["stimulus"].data:
        if stimulus in reps:
            reps[stimulus] += 1
        else:
            reps[stimulus] = 0
        repetitions.append(reps[stimulus])
    return assembly.assign_coords(
        {"repetition": ("presentation", np.array(repetitions).astype(np.uint8))},
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
            "stimulus": [path.stem for path in paths],
            "filename": paths,
        },
    ).set_index("stimulus")
