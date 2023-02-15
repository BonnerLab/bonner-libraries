from pathlib import Path

import pandas as pd
from PIL import Image
from scipy.io import loadmat


def save_images() -> list[Path]:
    images = loadmat("images_natimg2800_all.mat", simplify_cells=True)["imgs"]
    Path("images").mkdir(parents=True, exist_ok=True)
    paths = []
    for i_image in range(images.shape[-1]):
        path = Path("images") / f"image{i_image:04}.png"
        paths.append(path)
        Image.fromarray(images[:, :, i_image]).save(path)
    return paths


def create_stimulus_set() -> pd.DataFrame:
    paths = save_images()
    return pd.DataFrame(
        {
            "stimulus_id": [path.stem for path in paths],
            "filename": paths,
        }
    )
