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

IDENTIFIER = "hebart2022.things.behavior"
PROJECT_ID = "f5rn6"

CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def _download_osf_project(project_id, save_path, target_paths=None, use_cached=True):
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')
    
    if (not use_cached) or (not save_path.exists()):
        os.makedirs(save_path, exist_ok=True)
        for file in storage.files:
            if target_paths is not None and file.path not in target_paths:
                continue
            file_path = os.path.join(save_path, file.path.lstrip('/'))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as local_file:
                file.write_to(local_file)
            
            if file_path.endswith('.zip'):
                file_path = unzip(Path(file_path), extract_dir=save_path)


def load_embeddings():
    _download_osf_project(
        project_id=PROJECT_ID,
        save_path=CACHE_PATH,
        target_paths=(
            "/data/spose_embedding_66d_sorted.txt",
            "/variables/labels.txt",
            "/variables/unique_id.txt",
        ),
    )
    
    embd = pd.read_csv(CACHE_PATH / "data" / "spose_embedding_66d_sorted.txt", sep="\t", header=None).values
    bhv = pd.read_csv(CACHE_PATH / "variables" / "labels.txt", sep="\t", header=None).values.flatten()
    object = pd.read_csv(CACHE_PATH / "variables" / "unique_id.txt", sep="\t", header=None).values.flatten()
    
    return xr.DataArray(
        embd,
        dims=("object", "behavior"),
        coords={"object": object, "behavior": bhv},
    )
    
