import json
from pathlib import Path

from PIL import Image
import pandas as pd
import xarray as xr
from torchdata.datapipes.map import MapDataPipe

from bonner.files import download_from_s3, download_from_url, unzip
from bonner.datasets.allen2021_natural_scenes._utilities import (
    BUCKET_NAME,
    CACHE_PATH,
    IDENTIFIER,
)
from bonner.datasets._utilities import BONNER_DATASETS_HOME

N_STIMULI = 73000


def download_captions(force_download: bool = False) -> Path:
    URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    directory = BONNER_DATASETS_HOME / "coco_captions"
    filepath = download_from_url(
        URL,
        filepath=directory / "annotations_trainval2017.zip",
        force=force_download,
    )
    unzip(filepath, extract_dir=directory, remove_zip=False)
    return directory / "annotations"


def load_captions() -> pd.DataFrame:
    def _helper(subset: str = "train") -> pd.DataFrame:
        directory = download_captions()
        match subset:
            case "train" | "val":
                with open(directory / f"captions_{subset}2017.json", "r") as f:
                    annotations = json.load(f)["annotations"]
            case _:
                raise ValueError()

        result = {}
        for annotation in annotations:
            coco_id = annotation["image_id"]
            if coco_id not in result:
                result[coco_id] = []

            result[coco_id].append(annotation["caption"])
        return pd.DataFrame.from_dict(
            {"cocoId": result.keys(), "captions": result.values()}
        ).set_index("cocoId")

    return pd.concat(
        [_helper(subset).assign(subset=subset) for subset in ("train", "val")],
        axis=0,
    )


def load_stimulus_metadata() -> pd.DataFrame:
    filepath = Path("nsddata") / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    metadata = (
        pd.read_csv(CACHE_PATH / filepath, sep=",")
        .rename(columns={"Unnamed: 0": "stimulus_id"})
        .assign(
            stimulus_id=lambda x: "image"
            + x["stimulus_id"].astype("string").str.zfill(5)
        )
        .set_index("stimulus_id")
    )
    return pd.merge(
        metadata, load_captions(), left_on="cocoId", right_index=True
    )


def load_stimuli() -> xr.DataArray:
    filepath = Path("nsddata_stimuli") / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return (
        xr.open_dataset(CACHE_PATH / filepath)["imgBrick"]
        .rename(
            {
                "phony_dim_0": "stimulus_id",
                "phony_dim_1": "height",
                "phony_dim_2": "width",
                "phony_dim_3": "channel",
            }
        )
        .assign_coords(
            {
                "channel": ("channel", ["R", "G", "B"]),
                "stimulus_id": (
                    "stimulus_id",
                    [f"image{idx:05}" for idx in range(N_STIMULI)],
                ),
            }
        )
    )


class StimulusSet(MapDataPipe):
    def __init__(self) -> None:
        self.identifier = IDENTIFIER
        self.metadata = load_stimulus_metadata()
        self.stimuli = load_stimuli()

    def __getitem__(self, stimulus_id: str) -> Image.Image:
        return Image.fromarray(self.stimuli.sel(stimulus_id=stimulus_id).values)

    def __len__(self) -> int:
        return len(self.metadata.index)
