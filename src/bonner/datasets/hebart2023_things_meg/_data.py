from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import requests
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import unzip
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

file_id = 39472855     # From the provided URL



save_path = "THINGS_data_MEG_preprocessed_dataset.zip"


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
    for file_info in article_data['files']:
        if file_info['id'] == file_id:
            file_url = file_info['download_url']
            break
    
    if file_url is None:
        logging.info(f"File with ID {file_id} not found in article {article_id}")
        return
    
    # Download the file
    file_response = requests.get(file_url, stream=True)
    file_response.raise_for_status()

    # Save the file to the specified location
    with open(save_path, 'wb') as file:
        for chunk in file_response.iter_content(chunk_size=8192):
            file.write(chunk)
            
    if save_path.endswith('.zip'):
        unzip(Path(save_path), extract_dir=save_path)
        

def download_dataset(preprocess_type: str = "preprocessed"):
    assert preprocess_type in ["preprocessed", "raw"], f"Invalid data type: {preprocess_type}"
    
    _download_figshare_file(
        ARTICLE_ID_DICT[preprocess_type],
        FILE_ID_DICT[preprocess_type],
    )
    