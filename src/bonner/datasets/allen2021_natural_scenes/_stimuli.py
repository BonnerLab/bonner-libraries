import itertools
import json
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from torch.utils.data import MapDataPipe
from tqdm import tqdm

from bonner.caching import cache
from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.datasets.allen2021_natural_scenes._utilities import (
    BUCKET_NAME,
    CACHE_PATH,
    IDENTIFIER,
    N_SUBJECTS,
)
from bonner.files import download_from_s3, download_from_url, unzip

N_STIMULI = 73000
N_OBJECT_CATEGORIES = 91
N_STUFF_CATEGORIES = 91


def download_annotations(*, force: bool = False) -> Path:
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "stuff_annotations": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
    }

    directory = BONNER_DATASETS_HOME / "coco"
    for label, url in urls.items():
        filepath = download_from_url(
            url,
            filepath=directory / f"{label}_trainval2017.zip",
            force=force,
        )
        unzip(filepath, extract_dir=directory, remove_zip=False)

    return directory / "annotations"


def get_coco_to_nsd_mapping() -> dict[int, int]:
    nsd = load_nsd_metadata()
    return {
        coco_id: nsd_id
        for nsd_id, coco_id in zip(
            int(nsd["nsdId"].to_numpy()),
            int(nsd["cocoId"].to_numpy()),
            strict=True,
        )
    }


@cache("metadata/annotations.nc", path=CACHE_PATH)
def load_annotations() -> xr.DataArray:
    directory = download_annotations()
    mapping = get_coco_to_nsd_mapping()

    n_instances = np.zeros(
        (N_STIMULI, N_OBJECT_CATEGORIES + N_STUFF_CATEGORIES + 1),
        dtype=np.uint8,
    )

    categories: list[pd.DataFrame] | pd.DataFrame = []
    for annotation_type in ("instances", "stuff"):
        for subset in ("train", "val"):
            with (directory / f"{annotation_type}_{subset}2017.json").open() as f:
                annotations = json.load(f)

            categories.append(
                pd.DataFrame.from_dict(annotations["categories"])
                .rename(columns={"id": "category_id"})
                .assign(
                    category_id=lambda x: x["category_id"] - 1,
                    annotation_type=annotation_type,
                )
                .set_index("category_id"),
            )

            for annotation in tqdm(
                annotations["annotations"],
                desc="annotation",
                leave=False,
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

    return xr.DataArray(
        name="count",
        data=n_instances[:, ~(n_instances == 0).all(axis=0)],
        dims=("stimulus", "category"),
        coords={
            "stimulus": np.arange(len(n_instances), dtype=np.uint32),
            "category": categories["name"].to_numpy().astype(str),
            "supercategory": (
                "category",
                categories["supercategory"].to_numpy().astype(str),
            ),
            "type": ("category", categories["annotation_type"].to_numpy().astype(str)),
        },
    )


@cache("metadata/captions.pkl", path=CACHE_PATH)
def load_captions() -> pd.DataFrame:
    mapping = get_coco_to_nsd_mapping()

    directory = download_annotations()

    captions: dict[int, list[str]] = {}

    for subset in ("train", "val"):
        with (directory / f"captions_{subset}2017.json").open() as f:
            annotations = json.load(f)["annotations"]

        for annotation in annotations:
            try:
                nsd_id = mapping[annotation["image_id"]]
            except:
                continue

            if nsd_id not in captions:
                captions[nsd_id] = []

            captions[nsd_id].append(annotation["caption"])

    return captions


def load_nsd_metadata() -> pd.DataFrame:
    filepath = Path("nsddata") / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return (
        pd.read_csv(
            CACHE_PATH / filepath,
            sep=",",
            dtype={
                f"subject{subject}_rep{rep}": np.uint32
                for (subject, rep) in itertools.product(
                    range(1, 1 + N_SUBJECTS),
                    range(3),
                )
            }
            | {f"subject{subject}": bool for subject in range(1, 1 + N_SUBJECTS)}
            | {
                "nsdId": np.uint32,
                "cocoId": np.uint64,
            },
        )
        .rename(
            columns={
                "Unnamed: 0": "stimulus",
            }
            | {
                f"subject{subject + 1}_rep{rep}": f"subject{subject}_rep{rep}"
                for (subject, rep) in itertools.product(range(N_SUBJECTS), range(3))
            }
            | {
                f"subject{subject + 1}": f"subject{subject}"
                for subject in range(N_SUBJECTS)
            },
        )
        .assign(stimulus=lambda x: x["stimulus"].astype(np.uint32))
        .set_index("stimulus")
    )


def load_stimuli() -> xr.DataArray:
    filepath = Path("nsddata_stimuli") / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    download_from_s3(filepath, bucket=BUCKET_NAME, local_path=CACHE_PATH / filepath)
    return (
        xr.open_dataset(CACHE_PATH / filepath)["imgBrick"]
        .rename(
            {
                "phony_dim_0": "stimulus",
                "phony_dim_1": "height",
                "phony_dim_2": "width",
                "phony_dim_3": "channel",
            },
        )
        .assign_coords(
            {
                "channel": ("channel", ["R", "G", "B"]),
                "stimulus": (
                    "stimulus",
                    np.arange(N_STIMULI, dtype=np.uint32),
                ),
            },
        )
        .transpose("stimulus", "channel", "height", "width")
        .rename("stimuli")
    )


class StimulusSet(MapDataPipe):
    def __init__(self: Self) -> None:
        self.identifier = IDENTIFIER
        self.stimuli = load_stimuli()
        self.annotations = load_annotations()
        self.captions = load_captions()

    def __getitem__(self: Self, stimulus: int) -> Image.Image:
        return Image.fromarray(
            self.stimuli.sel(stimulus=stimulus)
            .transpose("height", "width", "channel")
            .to_numpy(),
        )

    def __len__(self: Self) -> int:
        return self.stimuli.sizes["stimulus"]
