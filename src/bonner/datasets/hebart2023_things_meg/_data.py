from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import mne
# mne.set_log_level('ERROR')
import requests
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from joblib import Parallel, delayed

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import untar

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
N_SESSIONS = 12
N_JOBS = 6
FREQ = 1200
TRIGGER_AMPLITUDE = 64
TRIGGER_CHANNEL = "UPPT001"



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

### _from things meg ###
def setup_paths(meg_dir, session):
    run_paths,event_paths = [],[]
    for file in os.listdir(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/'):
        if file.endswith(".ds") and file.startswith("sub"):
            run_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
        if file.endswith("events.tsv") and file.startswith("sub"):
            event_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
    run_paths.sort()
    event_paths.sort()
    return run_paths, event_paths

def read_raw(curr_path,session,run,participant):
    raw = mne.io.read_raw_ctf(curr_path, preload=True, verbose=False)
    # signal dropout in one run -- replacing values with median
    if participant == '1' and session == 11 and run == 4:  
        n_samples_exclude = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-13.4)):np.argmin(np.abs(raw.times-13.4))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T
    elif participant == '2' and session == 10 and run == 2: 
        n_samples_exclude = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-59.8)):np.argmin(np.abs(raw.times-59.8))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T

    raw.drop_channels('MRO11-1609')
        
    return raw    

def read_events(event_paths,run,raw):
    # load event file that has the corrected onset times (based on optical sensor and replace in the events variable)
    event_file = pd.read_csv(event_paths[run],sep='\t')
    event_file.fillna(999999,inplace=True)
    events = mne.find_events(raw, stim_channel=TRIGGER_CHANNEL,initial_event=True, verbose=False)
    events = events[events[:,2]==TRIGGER_AMPLITUDE]
    # not useful at all...
    # events[:,0] = event_file['onset'] * 1000
    return events, event_file

def concat_epochs(raw, events, event_file, epochs, tmin, tmax):
    if epochs:
        epochs_1 = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, picks='mag', baseline=None, verbose=False, preload=True,)
        epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
        epochs_1.metadata = event_file[["file_path", "trial_type"]]
        epochs = mne.concatenate_epochs([epochs, epochs_1,], verbose=False)
    else:
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, picks='mag', baseline=None, verbose=False, preload=True,)
        epochs.metadata = event_file[["file_path", "trial_type"]]
    return epochs

def baseline_correction(epochs, baseline):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(copy=False), times=epochs.times, baseline=baseline, mode='zscore', copy=False, verbose=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin, event_id=epochs.event_id, verbose=False)
    return epochs
    
def run_preprocessing(meg_dir,session,participant, l_freq, h_freq, tmin, tmax, baseline, downsample_freq,):
    epochs = []
    run_paths, event_paths = setup_paths(meg_dir, session)
    for run, curr_path in enumerate(run_paths):
        raw = read_raw(curr_path, session, run, participant)
        events, event_file = read_events(event_paths, run, raw)
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        epochs = concat_epochs(raw, events, event_file, epochs, tmin, tmax)
        del raw
        epochs.drop_bad()
    metadata = epochs.metadata
    if baseline:
        epochs = baseline_correction(epochs, baseline)
    if downsample_freq < FREQ:
        epochs = epochs.resample(sfreq=downsample_freq)
    epochs.metadata = metadata
    return epochs
######################
    
def load_preprocessed_data(
    subject: int,
    data_type: str = "test",
    from_raw: bool = False,
    downsample_freq: int = 200,
    l_freq: float = 0.1,
    h_freq: float = 40,
    tmin: float = -0.1,
    tmax: float = 1.3,
    baseline: set[float, float] = None,
    rois: (str | list[str]) = None,
    return_epochs: bool = False,
    tfr_n_bin: int = None,
    band_stop_n_bin: int = None,
    average: bool = False,
    **kwargs
) -> xr.DataArray:
    if not from_raw:
        download_dataset(preprocess_type="preprocessed")
        data = mne.read_epochs(CACHE_PATH / "preprocessed" / "LOCAL/ocontier/thingsmri/openneuro/THINGS-data/THINGS-MEG/ds004212/derivatives/preprocessed" / 
        f"preprocessed_P{subject:01d}-epo.fif", preload=True, verbose=False)
        metadata = data.metadata
        if downsample_freq < FREQ:
            data = data.resample(sfreq=downsample_freq)
        data.metadata = metadata
        path_column = "image_path"
    else:
        raw_path = CACHE_PATH / "raw" / "THINGS-MEG"
        download_dataset(preprocess_type="raw")
        meg_dir = raw_path / f"sub-BIGMEG{subject}/"
        
        data = Parallel(n_jobs=N_JOBS, backend="multiprocessing")(delayed(run_preprocessing)(meg_dir, session, subject, l_freq, h_freq, tmin, tmax, baseline, downsample_freq,) for session in range(1,N_SESSIONS+1))
        for epochs in data:
            epochs.info['dev_head_t'] = data[0].info['dev_head_t']
        data = mne.concatenate_epochs(epochs_list=data, add_offset=True)
        path_column = "file_path"

    if rois is not None:
        if isinstance(rois, str):
            rois = [rois]
        rois = np.concatenate([ROI_DICT[r] for r in rois])
        channels = np.array(data.ch_names)
        channels = channels[np.array([ch[:3] in rois for ch in channels])]
        data = data.pick(channels)
        
    if tfr_n_bin is not None:
        freqs = np.logspace(np.log10(4), np.log10(h_freq), tfr_n_bin)
        data = data.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=freqs / 2.0,
            n_jobs=N_JOBS,
            average=average,
        )
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
        band_stop_data = []
        for i in tqdm(range(len(freqs)), desc="freq"):
            temp = mne.filter.filter_data(
                data=data.get_data(copy=False),
                sfreq=data.info['sfreq'],
                l_freq=bin_boundaries[i+1],
                h_freq=bin_boundaries[i],
                n_jobs=-1,
                verbose=False,
                method="iir",
            )
            band_stop_data.append(temp)
    
    if return_epochs:
        #  until needed: in the case of band_stop_n_bin is no None, still return the original data
        return data
    
    metadata = data.metadata
    neuroid = data.ch_names
    times = data.times
    epoch_idx = metadata.trial_type == data_type
    metadata = metadata.loc[epoch_idx]
    
    if band_stop_n_bin is None:
        data = data.get_data(copy=False) if tfr_n_bin is None else data.get_data()
    else:
        data = np.stack(band_stop_data, axis=2)
        del band_stop_data
    data = data[epoch_idx]
    
    # temporary fix for time digit fix
    times = np.round(times, 3)
    
    img_files = [
        i.split('/')[-1]
        for i in metadata[path_column]
    ]
    object = [
        '_'.join(i.split('_')[:-1])
        for i in img_files
    ]
    if tfr_n_bin is None and band_stop_n_bin is None:
        data = xr.DataArray(
            data,
            dims=("object", "neuroid", "time"),
            coords={
                "object": object,
                "neuroid": neuroid,
                "time": times,
            },
        )
    else:
        data = xr.DataArray(
            data,
            dims=("object", "neuroid", "freq", "time"),
            coords={
                "object": object,
                "neuroid": neuroid,
                "freq": freqs,
                "time": times,
            },
        )
    
    data = data.assign_coords({"img_files": ("object", img_files)})
    return data
    
    