from PIL import Image
import pandas as pd
from torchdata.datapipes.map import MapDataPipe

from bonner.files import download_from_url, unzip
from bonner.datasets._utilities import BONNER_DATASETS_HOME

IDENTIFIER = "hebart2019.things"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
URL = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip="


def get_password() -> bytes:
    with open(CACHE_PATH / "password.txt", "r") as f:
        text = f.read()
    return text.split(" ")[-1].encode()


def download_stimuli(force: bool = False) -> None:
    path = CACHE_PATH / "download" / "things.zip"
    download_from_url(URL, filepath=path, force=force)

    password = get_password()

    unzip(path, extract_dir=CACHE_PATH, password=password, remove_zip=False)
    path = CACHE_PATH / "THINGS" / "Images"
    for suffix in ("A-C", "D-K", "L-Q", "R-S", "T-Z"):
        unzip(
            path / f"object_images_{suffix}.zip",
            extract_dir=CACHE_PATH / "images",
            remove_zip=False,
            password=password,
        )


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(
        CACHE_PATH / "THINGS" / "Metadata" / "Image-specific" / "image_paths.csv",
        sep=",",
        header=None,
        index_col=None,
        names=["filename"],
    )
    metadata["stimulus_id"] = [
        (CACHE_PATH / filename).stem for filename in metadata["filename"]
    ]
    return metadata.set_index("stimulus_id")


class StimulusSet(MapDataPipe):
    def __init__(self) -> None:
        download_stimuli()
        self.identifier = IDENTIFIER
        self.metadata = load_metadata()
        self.root = CACHE_PATH

    def __getitem__(self, index: str) -> Image.Image:
        return Image.open(self.root / self.metadata.loc[index, ["filename"]].item())

    def __len__(self) -> int:
        return len(self.metadata.index)
