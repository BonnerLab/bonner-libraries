from typing import Self

import pandas as pd
from PIL import Image
from torch.utils.data import MapDataPipe

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_url, unzip

IDENTIFIER = "hebart2019.things"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
URL = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip="

PASSWORD = "things4all"


def get_password() -> bytes:
    with (CACHE_PATH / "password.txt").open("r") as f:
        text = f.read()
    return text.split(" ")[-1].encode()


def download_stimuli(*, force: bool = False) -> None:
    path = CACHE_PATH / "download" / "things.zip"
    download_from_url(URL, filepath=path, force=force)

    unzip(path, extract_dir=CACHE_PATH, password=PASSWORD, remove_zip=False)
    unzip(
        CACHE_PATH / "_image_database_things.zip",
        extract_dir=CACHE_PATH / "images",
        remove_zip=False,
        password=PASSWORD.encode(),
    )


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(
        CACHE_PATH / "THINGS" / "Metadata" / "Image-specific" / "image_paths.csv",
        sep=",",
        header=None,
        index_col=None,
        names=["filename"],
    )
    metadata["stimulus"] = [
        (CACHE_PATH / filename).stem for filename in metadata["filename"]
    ]
    return metadata.set_index("stimulus")


class StimulusSet(MapDataPipe):
    def __init__(self: Self) -> None:
        download_stimuli()
        self.identifier = IDENTIFIER
        self.metadata = load_metadata()
        self.root = CACHE_PATH

    def __getitem__(self: Self, index: str) -> Image.Image:
        return Image.open(self.root / self.metadata.loc[index, ["filename"]].item())

    def __len__(self: Self) -> int:
        return len(self.metadata.index)
