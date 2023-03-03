from pathlib import Path

from PIL import Image
import pandas as pd
import xarray as xr
from torchdata.datapipes.map import MapDataPipe

from bonner.files import download_from_s3
from bonner.datasets.allen2021_natural_scenes._utilities import (
    BUCKET_NAME,
    CACHE_PATH,
    IDENTIFIER,
)

N_STIMULI = 73000


def load_stimulus_metadata() -> pd.DataFrame:
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
        identifier = IDENTIFIER
        metadata = load_stimulus_metadata()
        super().__init__(identifier=identifier, metadata=metadata)

    def __getitem__(self, index: str) -> Image.Image:
        return Image.fromarray(load_stimuli().sel(stimulus_id=index).values)

    def __len__(self) -> int:
        return len(self.metadata.index)
