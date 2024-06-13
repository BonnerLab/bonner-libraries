from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_s3

IDENTIFIER = "grootswagers2022.things_eeg"
BUCKET_NAME = "openneuro.org"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 50

def _download_data():
    """Download the data from the Grootswagers et al. (2022) THINGS EEG dataset."""
    s3_path = Path("ds003825 ds003825-download")
    download_from_s3(
        s3_path=s3_path,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH,
    )
    
def load_data():
    pass

def load_events():
    pass

def load_stimuli():
    pass