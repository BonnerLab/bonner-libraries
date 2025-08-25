from typing import Self

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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
        CACHE_PATH / "01_image-level" / "image-paths.csv",
        sep=",",
        header=None,
        index_col=None,
        names=["filename"],
    )
    metadata["stimulus"] = [
        (CACHE_PATH / filename).stem for filename in metadata["filename"]
    ]
    metadata = metadata.drop(columns="filename")
    metadata = pd.concat(
        [
            metadata,
            metadata["stimulus"]
            .str.rsplit("_", expand=True, n=1)
            .rename(
                columns={0: "concept", 1: "instance"},
            ),
        ],
        axis=1,
    )
    metadata["concept"] = metadata["concept"].astype(pd.CategoricalDtype())
    metadata["reference"] = [
        "b" in instance for instance in metadata["instance"].tolist()
    ]
    metadata["imagenet"] = [
        "n" in instance for instance in metadata["instance"].tolist()
    ]
    metadata["index"] = metadata["instance"].str[:2].astype(int) - 1
    metadata.attrs = {
        "reference": "whether this image was used as the reference image for the concept, and as part of the original triplet odd-one-out task",
        "imagenet": "whether this image is part of the ImageNet database",
    }
    return metadata


class StimulusSet(Dataset):
    def __init__(self: Self) -> None:
        download_stimuli()
        self.identifier = IDENTIFIER
        self.metadata = load_metadata()
        self.root = CACHE_PATH

    def __getitem__(self: Self, idx: int) -> Image.Image:
        row = self.metadata.iloc[idx]
        filename, concept = row["stimulus"], row["concept"]
        return Image.open(
            self.root / "images" / "object_images" / f"{concept}" / f"{filename}.jpg",
        )

    def __len__(self: Self) -> int:
        return len(self.metadata.index)
