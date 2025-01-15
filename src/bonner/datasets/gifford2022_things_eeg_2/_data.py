from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import mne
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import Counter

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import unzip
from osfclient.api import OSF
import requests

IDENTIFIER = "gifford2022.things_eeg_2"
PROJECT_ID_DICT = {
    "preprocessed": "anp5v",
    "images": "y63gw",
}
ARTICLE_ID_DICT = {
    "raw": 18470912,
}
METADATA_COLUMNS = ["img_files", "img_concepts", "img_concepts_THINGS"]
TYPE_DICT = {"train": "training", "test": "test"}
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 10
N_SESSIONS = 4
FREQ = 1000
L_FREQ, H_FREQ = 0.1, 100
SEED = 11
N_JOBS = 6


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

def _download_figshare_article(article_id, save_path, use_cached=True):
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
        file_url = file_info['download_url']
        file_name = Path(CACHE_PATH / file_info['name'])
        
        os.makedirs(file_name.parent, exist_ok=True)
        
        # Download the file
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()

        # Save the file to the specified location
        with open(file_name, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        if str(file_name).endswith('.zip'):
            unzip(CACHE_PATH / file_name, extract_dir=save_path)

def download_dataset(preprocess_type: str = "preprocessed"):
    match preprocess_type:
        case "preprocessed":
            _download_osf_project(
                project_id=PROJECT_ID_DICT[preprocess_type],
                save_path=CACHE_PATH / preprocess_type
            )
        case "raw":
            _download_figshare_article(
                article_id=ARTICLE_ID_DICT[preprocess_type],
                save_path=CACHE_PATH / preprocess_type
            )
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

def baseline_correction(epochs, baseline):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(copy=False), times=epochs.times, baseline=baseline, mode='zscore', copy=False, verbose=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin, event_id=epochs.event_id, verbose=False)
    return epochs
   
### adapted from things eeg 2 ###
def run_preprocessing(subject, data_type, downsample_freq, l_freq, h_freq, tmin, tmax, baseline, tfr_n_bin, band_stop_n_bin, band_stop, rois):
    epoched_data = []
    img_conditions = []
    events_list = []
    for session in range(1, N_SESSIONS+1):
        raw_path = CACHE_PATH / "raw" / f"sub-{subject:02d}" / f"ses-{session:02d}" / f"raw_eeg_{TYPE_DICT[data_type]}.npy"
        
        eeg_data = np.load(raw_path, allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        del eeg_data
        
        if l_freq > L_FREQ:
            raw.filter(l_freq=l_freq, h_freq=None, verbose=False)
        if h_freq < H_FREQ:
            raw.filter(l_freq=None, h_freq=h_freq, verbose=False)

        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        match rois:
            case "op":
                # Select only occipital (O) and posterior (P) channels
                chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'], '^O *|^P *'))
                new_chans = [raw.info['ch_names'][c] for c in chan_idx]
                raw.pick(new_chans)
            case "all":
                pass
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        ### Epoching, baseline correction and resampling ###
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        if baseline:
            epochs = baseline_correction(epochs, baseline)
        del raw
        # Resampling
        if downsample_freq < FREQ:
            epochs.resample(downsample_freq, verbose=False)
        
        if band_stop is not None:
            epochs = epochs.filter(
                l_freq=band_stop[1],
                h_freq=band_stop[0], 
                n_jobs=-1,
                verbose=False,
                method="iir",
            )
            
        if tfr_n_bin is not None:
            freqs = np.logspace(np.log10(4), np.log10(h_freq), tfr_n_bin)
            epochs = epochs.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=freqs / 2.0,
                n_jobs=N_JOBS,
                verbose=False,
            )
            data = epochs.get_data()
        elif band_stop_n_bin is not None:
            freqs = np.logspace(np.log10(4), np.log10(h_freq), band_stop_n_bin)
            log_freqs = np.log10(freqs)
            bin_width = (log_freqs[1] - log_freqs[0]) / 2
            log_bin_boundaries = np.concatenate([
                [log_freqs[0] - bin_width],
                (log_freqs[:-1] + log_freqs[1:]) / 2,
                [log_freqs[-1] + bin_width]
            ])
            bin_boundaries = 10**log_bin_boundaries
            data = []
            for i in tqdm(range(len(freqs)), desc="freq"):
                temp = mne.filter.filter_data(
                    data=epochs.get_data(copy=False),
                    sfreq=epochs.info['sfreq'],
                    l_freq=bin_boundaries[i+1],
                    h_freq=bin_boundaries[i],
                    n_jobs=-1,
                    verbose=False,
                    method="iir",
                )
                data.append(temp)
            data = np.stack(data, axis=2)
        else:
            data = epochs.get_data(copy=False)
        
        ch_names = epochs.info['ch_names']
        times = epochs.times

        # epoched_data.append(data)
        # events = epochs.events[:,2]
        # events_list.append(events)
        # img_cond = np.unique(events)
        # img_conditions.append(img_cond)
        ### Sort the data ###
        events = epochs.events[:,2]
        img_cond = np.unique(events)
        del epochs
        # Select only a maximum number of EEG repetitions
        if data_type == 'test':
            max_rep = 20
        else:
            max_rep = 2
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,*data.shape[1:]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=SEED, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)
        del sorted_data
    
    # return epoched_data, events_list, img_conditions
    
    epoched_data = np.concatenate(epoched_data, axis=1)
    # idx = shuffle(np.arange(0, epoched_data.shape[1]), random_state=SEED)
    # epoched_data = epoched_data[:,idx]
    
    mean = np.mean(epoched_data, axis=(0, 1), keepdims=True)
    std = np.std(epoched_data, axis=(0, 1), keepdims=True)
    epoched_data = (epoched_data - mean) / std
    
    return {
        'preprocessed_eeg_data': epoched_data,
        'ch_names': ch_names,
        'times': times
    }
    
    
def load_preprocessed_data(
    subject: int,
    data_type: str = "train",
    from_raw: bool = False,
    downsample_freq: int = 100,
    l_freq: float = 0.1,
    h_freq: float = 50,
    tmin: float = -.2,
    tmax: float = .8,
    baseline: set[float, float] = None,
    tfr_n_bin: int = None,
    band_stop_n_bin: int = None,
    band_stop: list[float, float] = None,
    rois: str = "op",
    **kwargs,
) -> tuple[xr.DataArray, pd.DataFrame]:
    if tfr_n_bin is not None or band_stop_n_bin is not None or band_stop is not None:
        assert from_raw
    
    if not from_raw:
        download_dataset(preprocess_type="preprocessed")
        x = np.load(CACHE_PATH / "preprocessed" / f"sub-{subject:02d}" / f"preprocessed_eeg_{TYPE_DICT[data_type]}.npy", allow_pickle=True).item()
    else:
        download_dataset(preprocess_type="raw")
        x = run_preprocessing(
            subject, data_type, downsample_freq, l_freq, h_freq, tmin, tmax, baseline, tfr_n_bin, band_stop_n_bin, band_stop, rois,
        )
    
    metadata = load_metadata(data_type=data_type)
    object = ["_".join(metadata.loc[i, METADATA_COLUMNS[1]].split("_")[1:]) for i in range(len(metadata))]
    # temporary fix for time digit fix
    times = np.round(x["times"], 2)
    
    if tfr_n_bin is None and band_stop_n_bin is None:
        data = xr.DataArray(
            x["preprocessed_eeg_data"],
            dims=("object", "presentation", "neuroid", "time"),
            coords={
                "object": object,
                "neuroid": x["ch_names"],
                "time": times,
            },
        )
    else:
        data = xr.DataArray(
            x["preprocessed_eeg_data"],
            dims=("object", "presentation", "neuroid", "freq", "time"),
            coords={
                "object": object,
                "neuroid": x["ch_names"],
                "time": times,
                "freq": np.logspace(np.log10(4), np.log10(h_freq), tfr_n_bin) if tfr_n_bin is not None else np.logspace(np.log10(4), np.log10(h_freq), band_stop_n_bin),
            },
        )
    data = data.assign_coords({column: ("object", metadata[column]) for column in METADATA_COLUMNS})
    return data
   


def load_stimuli():
    pass
