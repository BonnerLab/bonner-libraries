from pathlib import Path
from typing import Self

import pandas as pd
import scipy
from PIL import Image
from torch.utils.data import Dataset

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_url, untar

IDENTIFIER = "khosla2021.stanford-dogs"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def download_dataset(*, force: bool = False) -> None:
    url_root = "http://vision.stanford.edu/aditya86/ImageNetDogs"
    filenames = "lists.tar", "images.tar"

    for filename in filenames:
        filepath = download_from_url(
            url=f"{url_root}/{filename}",
            filepath=CACHE_PATH / Path(filename),
            force=force,
        )
        untar(filepath, extract_dir=CACHE_PATH, remove_tar=False)


def load_metadata() -> pd.DataFrame:
    download_dataset()
    metadata = scipy.io.loadmat(
        BONNER_DATASETS_HOME / IDENTIFIER / "file_list.mat",
        simplify_cells=True,
    )
    metadata = pd.DataFrame(
        {
            key: value
            for key, value in metadata.items()
            if key in {"file_list", "labels"}
        },
    )
    metadata["labels"] -= 1

    def _parse(filename: str) -> tuple[str, str, str]:
        pieces = filename.split("-")[1:]
        category, filename = ("-".join(pieces)).split("/")
        synset, index = filename.split("_")
        index = int(index[:-4])
        return synset, category, index

    metadata = (
        pd.concat(
            [metadata, pd.DataFrame(metadata["file_list"].apply(_parse).tolist())],
            axis=1,
        )
        .drop(columns={"file_list", "labels"})
        .rename(
            columns={0: "synset", 1: "category", 2: "index"},
        )
    )
    metadata["category"] = metadata["category"].astype(pd.CategoricalDtype())
    metadata["index"] = metadata["index"].astype(int)
    return metadata.set_index(["category", "index"]).sort_index().reset_index()


class StimulusSet(Dataset):
    def __init__(self: Self) -> None:
        self.identifier = IDENTIFIER
        self.metadata = load_metadata()

    def __getitem__(self: Self, idx: int) -> Image.Image:
        metadata = dict(self.metadata.iloc[idx].items())
        filepath = (
            BONNER_DATASETS_HOME
            / IDENTIFIER
            / "Images"
            / f"{metadata['synset']}-{metadata['category']}"
            / f"{metadata['synset']}_{int(metadata['index'])}.jpg"
        )
        return Image.open(filepath)

    def __len__(self: Self) -> int:
        return self.stimuli.sizes["stimulus"]
