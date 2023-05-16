import json
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes.map import MapDataPipe

from bonner.files import download_from_s3, download_from_url, unzip
from bonner.datasets.allen2021_natural_scenes._utilities import (
    BUCKET_NAME,
    CACHE_PATH,
    IDENTIFIER,
)
from bonner.caching import cache
from bonner.datasets._utilities import BONNER_DATASETS_HOME

N_STIMULI = 73000
N_OBJECT_CATEGORIES = 91
N_STUFF_CATEGORIES = 91


def download_annotations(force_download: bool = False) -> Path:
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "stuff_annotations": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
    }

    for label, url in urls.items():
        directory = BONNER_DATASETS_HOME / "coco"
        filepath = download_from_url(
            url,
            filepath=directory / f"{label}_trainval2017.zip",
            force=force_download,
        )
        unzip(filepath, extract_dir=directory, remove_zip=False)

    return directory / "annotations"


@cache("bonner-datasets/allen2021_natural_scenes/instance-annotations.pkl")
def load_instances() -> Path:
    directory = download_annotations()
    nsd = load_nsd_metadata()
    mapping = {
        coco_id: nsd_id
        for nsd_id, coco_id in zip(nsd["nsdId"].values, nsd["cocoId"].values)
    }
    n_instances = np.full(
        fill_value=np.nan,
        shape=(N_STIMULI, N_OBJECT_CATEGORIES + N_STUFF_CATEGORIES + 1),
    )

    categories = []
    for annotation_type in ("instances", "stuff"):
        for subset in ("train", "val"):
            with open(directory / f"{annotation_type}_{subset}2017.json", "r") as f:
                annotations = json.load(f)

            categories.append(
                pd.DataFrame.from_dict(annotations["categories"])
                .rename(columns={"id": "category_id"})
                .assign(category_id=lambda x: x["category_id"] - 1)
                .set_index("category_id")
            )

            for annotation in tqdm(
                annotations["annotations"], desc="annotation", leave=False
            ):
                try:
                    nsd_id = mapping[annotation["image_id"]]
                except KeyError:
                    continue

                category_id = annotation["category_id"] - 1

                if np.isnan(n_instances[nsd_id, category_id]):
                    n_instances[nsd_id, category_id] = 0
                else:
                    n_instances[nsd_id, category_id] += 1

    categories = pd.concat(categories).drop_duplicates().sort_index()

    return (
        pd.DataFrame(
            n_instances[:, (~np.isnan(n_instances).all(axis=0))],
            columns=categories["name"].to_list(),
        )
        .assign(stimulus_id=nsd.index)
        .set_index("stimulus_id")
        .fillna(0)
        .astype(np.uint8)
    )


@cache("bonner-datasets/allen2021_natural_scenes/captions.pkl")
def load_captions() -> pd.DataFrame:
    def _helper(subset: str) -> pd.DataFrame:
        directory = download_annotations()
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


def load_nsd_metadata() -> pd.DataFrame:
    filepath = Path("nsddata") / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return (
        pd.read_csv(CACHE_PATH / filepath, sep=",")
        .rename(columns={"Unnamed: 0": "stimulus_id"})
        .assign(
            stimulus_id=lambda x: "image"
            + x["stimulus_id"].astype("string").str.zfill(5)
        )
        .set_index("stimulus_id")
    )


def load_stimulus_metadata() -> pd.DataFrame:
    return (
        load_nsd_metadata()
        .merge(load_captions(), left_on="cocoId", right_index=True)
        .merge(load_instances(), on="stimulus_id")
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
