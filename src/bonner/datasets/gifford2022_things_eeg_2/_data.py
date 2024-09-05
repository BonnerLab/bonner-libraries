from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import mne
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import unzip
from osfclient.api import OSF

IDENTIFIER = "gifford2022.things_eeg_2"
PROJECT_ID_DICT = {
    "preprocessed": "anp5v",
    "images": "y63gw",
}
METADATA_COLUMNS = ["img_files", "img_concepts", "img_concepts_THINGS"]
TYPE_DICT = {"train": "training", "test": "test"}



CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 10


def _download_osf_project(project_id, save_path, use_cached=True):
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')
    
    if (not use_cached) or (not save_path.exists()):
        os.makedirs(save_path, exist_ok=True)
        for file in storage.files:
            file_path = os.path.join(save_path, file.path.lstrip('/'))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as local_file:
                file.write_to(local_file)
            
            if file_path.endswith('.zip'):
                file_path = unzip(Path(file_path), extract_dir=save_path)


def download_dataset(preprocess_type: str = "preprocessed"):
    match preprocess_type:
        case "preprocessed":
            _download_osf_project(
                project_id=PROJECT_ID_DICT[preprocess_type],
                save_path=CACHE_PATH / preprocess_type
            )
        case "raw":
            pass
        case "source":
            pass
        case _:
            raise ValueError(f"Invalid data type: {preprocess_type}")
 
 
def load_metadata(data_type: str = "train",) -> pd.DataFrame:
    _download_osf_project(
        project_id=PROJECT_ID_DICT["images"],
        save_path=CACHE_PATH / "images"
    )
    
    metadata = np.load(CACHE_PATH / "images" / "image_metadata.npy", allow_pickle=True).item()
    return pd.DataFrame.from_dict({
        column: metadata[f"{data_type}_{column}"]
        for column in METADATA_COLUMNS
    })
    
def load_preprocessed_data(
    subject: int,
    downsample_freq: int = 100,
    data_type: str = "train",
    l_freq: float = None,
    h_freq: float = None,
    tmin: float = -0.2,
    tmax: float = .8,
    window_size: (int | float) = None,
    window_step: (int | float) = None,
    baseline: set[float, float] = None,
    scale: (str | float) = None
) -> tuple[xr.DataArray, pd.DataFrame]:
    if downsample_freq == 100:
        download_dataset(preprocess_type="preprocessed")
        data = np.load(CACHE_PATH / "preprocessed" / f"sub-{subject:02d}" / f"preprocessed_eeg_{TYPE_DICT[data_type]}.npy", allow_pickle=True).item()
        metadata = load_metadata(data_type=data_type)
        object = ["_".join(metadata.loc[i, METADATA_COLUMNS[1]].split("_")[1:]) for i in range(len(metadata))]
        # temporary fix for time digit fix
        times = np.round(data["times"], 2)
        
        data = xr.DataArray(
            data["preprocessed_eeg_data"],
            dims=("object", "presentation", "neuroid", "time"),
            coords={
                "object": object,
                "neuroid": data["ch_names"],
                "time": times,
            },
        )
        data = data.assign_coords({column: ("object", metadata[column]) for column in METADATA_COLUMNS})
        return data
    else:
        download_dataset(preprocess_type="raw")
        # TODO: implement method using raw-type data
        return None


def load_stimuli():
    pass
