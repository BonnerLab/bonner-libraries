from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Estimated head radius')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='The data contains')

import mne
import os
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_s3
from bonner.files import unzip
from osfclient.api import OSF

IDENTIFIER = "zhang2025.nod_eeg"
BUCKET_NAME = "openneuro.org"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 19
SUBJECT_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,26,27,29.30]


def download_dataset():
    """Download the whole dataset from its OpenNeuro S3 bucket into the local cache.

    Downloads the full accession, not a subject subset, and skips what is already present.
    """
    s3_path = Path("ds005811")
    download_from_s3(
        s3_path=s3_path,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH,
        is_dir=True
    )
    
