from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import mne
import requests
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import untar
from osfclient.api import OSF

IDENTIFIER = "hebart2023.things_meg"
ARTICLE_ID_DICT = {
    "preprocessed": 21215246,
    "raw": 20563800,
}
FILE_ID_DICT = {
    "preprocessed": 39472855,
    "raw": 36827316,
}
ROI_DICT = {
    "occipital": ["MLO", "MRO"],
    "temporal": ["MLT", "MRT"],
    "parietal": ["MLP", "MRP"],
    "frontal": ["MLF", "MRF"],
    "central": ["MLC", "MRC"],
}
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 4



def _download_figshare_file(article_id, file_id, save_path, use_cached=True):
    if use_cached and os.path.exists(save_path):
        return
    
    article_url = f"https://api.figshare.com/v2/articles/{article_id}"
    
    # Get the article metadata
    response = requests.get(article_url)
    response.raise_for_status()
    article_data = response.json()
    
    # Find the file in the article metadata
    file_url = None
    file_name = None
    for file_info in article_data['files']:
        if file_info['id'] == file_id:
            file_url = file_info['download_url']
            file_name = Path(CACHE_PATH / file_info['name'])
            break
    
    if file_url is None:
        logging.info(f"File with ID {file_id} not found in article {article_id}")
        return
    
    os.makedirs(file_name.parent, exist_ok=True)
    
    # Download the file
    file_response = requests.get(file_url, stream=True)
    file_response.raise_for_status()

    # Save the file to the specified location
    with open(file_name, 'wb') as file:
        for chunk in file_response.iter_content(chunk_size=8192):
            file.write(chunk)
            
    if str(file_name).endswith('.tar.gz'):
        untar(CACHE_PATH / file_name, extract_dir=save_path)
        

def download_dataset(preprocess_type: str = "preprocessed"):
    assert preprocess_type in ["preprocessed", "raw"], f"Invalid data type: {preprocess_type}"
    
    _download_figshare_file(
        ARTICLE_ID_DICT[preprocess_type],
        FILE_ID_DICT[preprocess_type],
        save_path=CACHE_PATH / preprocess_type
    )
    
    
def load_preprocessed_data(
    subject: int,
    downsample_freq: int = 200,
    data_type: str = "test",
    l_freq: float = None,
    h_freq: float = None,
    tmin: float = -0.1,
    tmax: float = 1.3,
    window_size: (int | float) = None,
    window_step: (int | float) = None,
    baseline: set[float, float] = None,
    scale: (str | float) = None,
    rois: (str | list[str]) = None,
) -> xr.DataArray:
    if downsample_freq == 200:
        download_dataset(preprocess_type="preprocessed")
        x = mne.read_epochs(CACHE_PATH / "preprocessed" / "LOCAL/ocontier/thingsmri/openneuro/THINGS-data/THINGS-MEG/ds004212/derivatives/preprocessed" / 
        f"preprocessed_P{subject:01d}-epo.fif", preload=True, verbose=False)
        metadata = x.metadata
        data = x.get_data()
        epoch_idx = metadata.trial_type == data_type
        metadata = metadata.loc[epoch_idx]
        data = data[epoch_idx]
        times = x.times
        # temporary fix for time digit fix
        # times = np.round(x.times, 3)
        
        img_files = [
            i.split('/')[-1]
            for i in metadata.image_path
        ]
        object = [
            '_'.join(i.split('_')[:-1])
            for i in img_files
        ]
        data = xr.DataArray(
            data,
            dims=("object", "neuroid", "time"),
            coords={
                "object": object,
                "neuroid": x.ch_names,
                "time": times,
            },
        )
        data = data.assign_coords({"img_files": ("object", img_files)})
        
        if rois is not None:
            data = data.sel(neuroid=roi_index(rois, data))
        return data
    else:
        download_dataset(preprocess_type="raw")
        # TODO: implement method using raw-type data
        return None
    

def roi_index(
    rois: (str | list[str]),
    X: xr.DataArray = load_preprocessed_data(subject=1),
) -> xr.DataArray:
    if isinstance(rois, str):
        rois = [rois]
    rois = np.concatenate([ROI_DICT[r] for r in rois])
    return [s[:3] in rois for s in X.neuroid.values]
    
    