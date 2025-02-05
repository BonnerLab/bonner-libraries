from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Estimated head radius')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='The data contains')

import mne
import numpy as np
import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_s3

IDENTIFIER = "grootswagers2022.things_eeg"
BUCKET_NAME = "openneuro.org"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 50
# SAMPLE_RATE = 1000
DOWNSAMPLE_RATE = 250
PRESENATION_DURATION = 50
N_STIM_MAIN = 22248
N_STIM_VALIDATION = 2400
EXCLUDED_SUBJECTS = [1, 6, 18, 23]



def download_dataset():
    """Download the data from the Grootswagers et al. (2022) THINGS EEG dataset."""
    s3_path = Path("ds003825")
    download_from_s3(
        s3_path=s3_path,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH,
        is_dir=True
    )


def load_preprocessed_data(
    subject: int,
    downsample_freq: int = 250,
    l_freq: float = None,
    h_freq: float = None,
    tmin: float = -0.1,
    tmax: float = 1.0,
    is_validation: bool = False,
    window_size: (int | float) = None,
    window_step: (int | float) = None,
    baseline: set[float, float] = None,
    scale: (str | float) = None
) -> tuple[xr.DataArray, pd.DataFrame]:
    # download_dataset()
    event_csv = pd.read_csv(CACHE_PATH / f"sub-{subject:02d}" / "eeg" / f"sub-{subject:02d}_task-rsvp_events.csv")
    if is_validation:
        if len(event_csv) != N_STIM_MAIN + N_STIM_VALIDATION:
            logging.warning("Validation data is not available for this subject.")
            return None, None
        
    x = mne.io.read_raw_eeglab(
        CACHE_PATH / "derivatives" / "eeglab" / f"sub-{subject:02d}_task-rsvp_continuous.set",
        preload=True, verbose=False,
    )
    
    if l_freq is not None:
        x.filter(l_freq=l_freq, h_freq=None, verbose=False)
    if h_freq is not None:
        x.filter(l_freq=None, h_freq=h_freq, verbose=False)
    if downsample_freq != DOWNSAMPLE_RATE:
        assert downsample_freq < DOWNSAMPLE_RATE
        x = x.resample(sfreq=downsample_freq, verbose=False)
        
    events, e_dict = mne.events_from_annotations(x, verbose=False)
    onset_idx = e_dict["E  1"]
    
    onset_ms = events[events[:, 2] == onset_idx, 0]
    if downsample_freq != DOWNSAMPLE_RATE:
        onset_ms = (onset_ms - 1) * (DOWNSAMPLE_RATE // downsample_freq) + 1
    onset_ms = onset_ms * 4 - 3
    df = pd.concat([
        event_csv,
        pd.DataFrame({
            "onset_ms": onset_ms, 
            "duration_ms": [PRESENATION_DURATION] * len(onset_ms),
        })
    ], axis=1,)
    
    epochs = mne.Epochs(
        x, events,
        event_id=onset_idx,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        verbose=False,
    )
    data = epochs.get_data()
    if scale is not None:
        if isinstance(scale, str):
            match scale:
                case "original":
                    data = data * 1e6
                case "std":
                    data = data / data.std()
        else:
            data = data / scale
    
    data = xr.DataArray(
        data=data,
        dims=("presentation", "neuroid", "time"),
        coords={
            "presentation": df["stimname"],
            "neuroid": epochs.ch_names,
            "time": epochs.times,
        }
    )
       
    if window_size is not None:
        assert window_step is not None
        dim_length = data.sizes["time"]
        if isinstance(window_size, float):
            window_size = int(window_size * dim_length)
        if isinstance(window_step, float):
            window_step = int(window_step * dim_length)
        indices = np.arange(0, dim_length, window_step)
        data = [
            (data
                .isel({"time": slice(i, i + window_size)})
                .mean(dim="time")
                .assign_coords({"time": data.time.values[i]})
            )
            for i in indices
        ]
        data = xr.concat(data, dim="time").transpose("presentation", "neuroid", "time")
    
    if is_validation:
        return data[N_STIM_MAIN:], df[N_STIM_MAIN:]
    else:
        return data[:N_STIM_MAIN], df[:N_STIM_MAIN]    


def load_stimuli():
    pass
