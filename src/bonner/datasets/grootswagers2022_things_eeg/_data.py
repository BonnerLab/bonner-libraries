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

IDENTIFIER = "grootswagers2022.things_eeg"
BUCKET_NAME = "openneuro.org"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 50
DOWNSAMPLE_RATE = 250
PRESENATION_DURATION = 50
N_STIM_MAIN = 22248
N_STIM_VALIDATION = 2400
EXCLUDED_SUBJECTS = [1, 6, 18, 23]
PROJECT_ID_DICT = {
    "codes": "e485y",
}



def download_dataset():
    """Download the Grootswagers et al. (2022) THINGS EEG dataset into the local cache."""
    s3_path = Path("ds003825")
    download_from_s3(
        s3_path=s3_path,
        bucket=BUCKET_NAME,
        local_path=CACHE_PATH,
        is_dir=True
    )
    
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
                
def load_metadata(data_type: str = "validation"):
    """Load the stimulus metadata: image filenames and the object concept each depicts.

    Only ``"validation"`` is implemented; any other value returns ``None``.

    Args:
    ----
        data_type: which split's metadata to load

    Returns:
    -------
        one row per image, with its filename and object concept

    """
    _download_osf_project(
        project_id=PROJECT_ID_DICT["codes"],
        save_path=CACHE_PATH / "codes"
    )
    match data_type:
        case "validation":
            img_files = pd.read_csv(CACHE_PATH / "codes" / "test_images.csv", header=None,)[0].values
            return pd.DataFrame({
                "img_files": img_files,
                "object": [s.split("/")[0] for s in img_files],
            })

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
    scale: (str | float) = "default",
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Load one subject's epoched EEG responses together with the events that produced them.

    The published recording is already downsampled, so ``downsample_freq`` can only lower the
    rate further and asserts rather than upsampling. Event onsets are rescaled to match whatever
    rate is requested, so they stay aligned with the returned epochs.

    Not every subject has the validation block; when ``is_validation`` is requested for a subject
    that lacks it, this warns and returns a pair of ``None`` rather than raising.

    Args:
    ----
        subject: subject number
        downsample_freq: target sampling rate; must not exceed the recorded rate
        l_freq: high-pass cutoff, or ``None`` for no high-pass
        h_freq: low-pass cutoff, or ``None`` for no low-pass
        tmin: epoch start relative to stimulus onset, in seconds
        tmax: epoch end relative to stimulus onset, in seconds
        is_validation: return the validation block rather than the main one
        window_size: width of the averaging window, if the epochs are to be binned over time
        window_step: stride between successive windows
        baseline: interval to baseline-correct against, or ``None`` for none
        scale: how to normalize the responses

    Returns:
    -------
        the responses and the event table describing the presentations they came from

    """
    download_dataset()
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
                case "default":
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
    """Not implemented — returns ``None``.

    The stimulus images are needed to pair these responses with model features, so this dataset
    cannot be used for that until this is written.
    """
    pass
