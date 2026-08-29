from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import re
import mne
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import MapDataPipe
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

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

# Maps rois value → (uses_old_data, channel_regex_or_None)
ROI_CONFIG = {
    "op":  (True, None),
    "o":   (True, r'^O'),
    "p":   (True, r'^P'),
    "all": (False, r'^(?!stim)'),
    "f":   (False, r'^F'),
    "c":   (False, r'^C'),
    "t":   (False, r'^T'),
}

def _roi_config(rois):
    if rois in ROI_CONFIG:
        return ROI_CONFIG[rois]
    pattern = '|'.join(f'^{c.upper()}' for c in rois)
    return (False, pattern)

def _filter_channels(x, channel_pattern):
    if channel_pattern is None:
        return x
    mask = [bool(re.match(channel_pattern, ch)) for ch in x["ch_names"]]
    x["ch_names"] = [ch for ch, m in zip(x["ch_names"], mask) if m]
    x["preprocessed_eeg_data"] = x["preprocessed_eeg_data"][:, :, mask, ...]
    return x


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

def _download_osf_files_filtered(project_id, save_path, file_filter=None, use_cached=True):
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')

    if (not use_cached) or (not save_path.exists()):
        os.makedirs(save_path, exist_ok=True)
        for file in storage.files:
            if file_filter is not None and not file_filter(file.path):
                continue
            file_path = os.path.join(save_path, file.path.lstrip('/'))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as local_file:
                file.write_to(local_file)

            if file_path.endswith('.zip'):
                unzip(Path(file_path), extract_dir=save_path)

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

def download_dataset(preprocess_type: str = "preprocessed", rois: str = "op"):
    """Download the dataset into the local cache.

    Args:
    ----
        preprocess_type: which release to fetch — the authors' preprocessed epochs, or the raw
            recordings that ``load_preprocessed_data`` can re-epoch itself
        rois: which channel selection to fetch

    """
    match preprocess_type:
        case "preprocessed":
            uses_old_data, _ = _roi_config(rois)
            if uses_old_data:
                _download_osf_files_filtered(
                    project_id=PROJECT_ID_DICT[preprocess_type],
                    save_path=CACHE_PATH / "preprocessed",
                    file_filter=lambda path: "63_channels" not in path,
                )
            else:
                _download_osf_files_filtered(
                    project_id=PROJECT_ID_DICT[preprocess_type],
                    save_path=CACHE_PATH / "preprocessed_all",
                    file_filter=lambda path: "63_channels" in path,
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
    """Load the stimulus metadata: image filenames and the object concept each depicts.

    Args:
    ----
        data_type: ``"train"``, ``"test"``, or ``"all"`` to concatenate both in the order the
            response arrays use

    Returns:
    -------
        one row per image

    """
    if data_type == "all":
        # Test before train: the concatenated index must match the order the epoched data
        # is stored in, so swapping these silently misaligns metadata against responses.
        return pd.concat([load_metadata("test"), load_metadata("train")], axis=0).reset_index(drop=True)
    else:
        # Kept, disabled, as the only in-repo record of how the image cache below is
        # populated: the load beneath it reads exactly what this call would have written.
        # _download_osf_project(
        #     project_id=PROJECT_ID_DICT["images"],
        #     save_path=CACHE_PATH / "images"
        # )
        
        metadata = np.load(CACHE_PATH / "images" / "image_metadata.npy", allow_pickle=True).item()
        df =  pd.DataFrame.from_dict({
            column: metadata[f"{data_type}_{column}"]
            for column in METADATA_COLUMNS
        })
        df["object"] = ['_'.join(s.split('_')[1:]) for s in df[METADATA_COLUMNS[1]].to_list()]
        return df
    
def load_stimuli(data_type: str = "train", batch_size: int = 256, idx: int = None) -> xr.DataArray:
    """Load the stimulus images, either all of them or a single one.

    Args:
    ----
        data_type: ``"train"``, ``"test"``, or ``"all"``
        batch_size: images loaded per batch when reading the whole set
        idx: return only this image, skipping the batched read

    Returns:
    -------
        images, with a ``stimulus`` dimension carrying the filenames

    """
    # Kept, disabled, as the only in-repo record of how the image cache below is populated:
    # the loader beneath it reads exactly what this call would have written.
    # _download_osf_project(
    #     project_id=PROJECT_ID_DICT["images"],
    #     save_path=CACHE_PATH / "images"
    # )
    
    stimuli_folder = CACHE_PATH / "images"
    
    if data_type != "all":
        stimuli_folder = stimuli_folder / f"{TYPE_DICT[data_type]}_images"
    
    dataset = ImageFolder(
        root=stimuli_folder,
        transform=ToTensor() 
    )
    
    if idx is None:
        # Load data into batches
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        all_images = []
        for images, _ in dataloader:
            all_images.append(images)
        
        all_images = torch.cat(all_images, dim=0).permute(0, 2, 3, 1)  # Shape: (N, H, W, C)
        
        return xr.DataArray(
            all_images.numpy(),
            dims=["stimulus", "height", "width", "channel",],
            coords={"stimulus": load_metadata(data_type)["img_files"].to_list()}
        )
    else:
        return xr.DataArray(
            dataset[idx][0].permute(1, 2, 0).unsqueeze(0),
            dims=["stimulus", "height", "width", "channel",],
            coords={"stimulus": [load_metadata(data_type)["img_files"][idx]]}
        )
    
class StimulusSet(MapDataPipe):
    """Indexable view of the stimulus images, yielding one PIL image per index.

    Pairs with the response loaders: index ``i`` here is the image described by row ``i`` of
    ``load_metadata`` for the same ``data_type``.
    """

    def __init__(self, data_type: str) -> None:
        self.data_type = data_type
        self.identifier = f"{IDENTIFIER}.{data_type}"
        self.metadata = load_metadata(data_type)
        
    def __getitem__(self, idx: int):
        return ToPILImage()(
            load_stimuli(data_type=self.data_type, idx=idx).isel(stimulus=0).values
        )

    def __len__(self) -> int:
        return len(self.metadata)

def baseline_correction(epochs, baseline):
    """Baseline-correct epochs by subtracting each channel's mean over the baseline interval.

    Args:
    ----
        epochs: the epochs to correct
        baseline: the interval to take the mean over

    Returns:
    -------
        the corrected epochs

    """
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(copy=False), times=epochs.times, baseline=baseline, mode='mean', copy=False, verbose=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin, event_id=epochs.event_id, verbose=False)
    return epochs

def run_preprocessing(subject, data_type, downsample_freq, l_freq, h_freq, tmin, tmax, baseline, tfr_n_bin, band_stop_n_bin, band_stop, rois, shuffle_reps=True):
    """Epoch and preprocess one subject's raw recordings, session by session.

    Sessions are processed independently and concatenated, with repetitions of each image
    gathered together so the result is indexed by image rather than by presentation.

    Args:
    ----
        subject: subject number
        data_type: ``"train"`` or ``"test"``
        downsample_freq: target sampling rate
        l_freq: high-pass cutoff
        h_freq: low-pass cutoff
        tmin: epoch start relative to stimulus onset, in seconds
        tmax: epoch end relative to stimulus onset, in seconds
        baseline: interval to baseline-correct against, or ``None`` for none
        tfr_n_bin: number of frequency bins, if a time-frequency representation is wanted
        band_stop_n_bin: number of band-stop bins
        band_stop: band to stop
        rois: channel selection
        shuffle_reps: shuffle repetitions of an image before they are stacked

    Returns:
    -------
        the epoched responses, the image conditions, and the event tables

    """
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

        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        
        match rois:
            case "op":
                # Select only occipital (O) and posterior (P) channels
                chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'], '^O *|^P *'))
                new_chans = [raw.info['ch_names'][c] for c in chan_idx]
                raw.pick(new_chans)
            case "f":
                chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'], '^F *'))
                new_chans = [raw.info['ch_names'][c] for c in chan_idx]
                raw.pick(new_chans)
            case "all":
                pass
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

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

        events = epochs.events[:,2]
        events_list.append(events)
        img_cond = np.unique(events)
        del epochs
        # Select only a maximum number of EEG repetitions
        if data_type == 'test':
            max_rep = 20
            sorted_data = np.zeros((len(img_cond),max_rep,*data.shape[1:]))
            for i in range(len(img_cond)):
                # Find the indices of the selected image condition
                idx = np.where(events == img_cond[i])[0]
                if shuffle_reps:
                    # Randomly select only the max number of EEG repetitions
                    idx = shuffle(idx, random_state=SEED, n_samples=max_rep)
                else:
                    # Keep chronological order: take first max_rep occurrences
                    idx = idx[:max_rep]
                sorted_data[i] = data[idx]
            del data
            epoched_data.append(sorted_data)
            del sorted_data
        else:
            # Unreachable by design. This branch mis-handles the concatenation and is kept
            # only to show the intended shape; the assertion stops a caller reaching it.
            assert False
            max_rep = 2
            if session % 2 == 1:
                # For odd sessions, select the first max_rep EEG repetitions
                sorted_data = np.zeros((len(img_cond)*2,max_rep,*data.shape[1:]))
                for i in range(len(img_cond)):
                    idx = np.where(events == img_cond[i])[0][:max_rep]
                    sorted_data[i] = data[idx]
                del data
            else:
                for i in range(len(img_cond)):
                    idx = np.where(events == img_cond[i])[0][:max_rep]
                    sorted_data[i] = data[idx]
                del data
                epoched_data.append(sorted_data)
                del sorted_data
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        img_conditions.append(img_cond)
       
    
    epoched_data = np.concatenate(epoched_data, axis=1)
    
    mean = np.mean(epoched_data, axis=(0, 1), keepdims=True)
    std = np.std(epoched_data, axis=(0, 1), keepdims=True)
    epoched_data = (epoched_data - mean) / std
    
    return {
        'preprocessed_eeg_data': epoched_data,
        'ch_names': ch_names,
        'times': times,
        # not useful as now the data is already shuffled
        'events_list': events_list,
        'img_conditions': img_conditions,
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
    shuffle: bool = True,
    **kwargs,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Load one subject's epoched responses, with the event table describing them.

    By default this reads the authors' preprocessed release. Set ``from_raw`` to re-epoch the raw
    recordings instead, which is what makes the filtering and epoching arguments below take
    effect — against the preprocessed release they are fixed at whatever the authors chose.

    Args:
    ----
        subject: subject number
        data_type: ``"train"`` or ``"test"``
        from_raw: re-epoch the raw recordings rather than reading the preprocessed release
        downsample_freq: target sampling rate
        l_freq: high-pass cutoff
        h_freq: low-pass cutoff
        tmin: epoch start relative to stimulus onset, in seconds
        tmax: epoch end relative to stimulus onset, in seconds
        baseline: interval to baseline-correct against, or ``None`` for none
        tfr_n_bin: number of frequency bins, if a time-frequency representation is wanted
        band_stop_n_bin: number of band-stop bins
        band_stop: band to stop
        rois: channel selection
        shuffle: shuffle repetitions of an image before they are stacked

    Returns:
    -------
        the responses and the event table describing the presentations they came from

    """
    if tfr_n_bin is not None or band_stop_n_bin is not None or band_stop is not None:
        assert from_raw

    if not from_raw:
        uses_old_data, channel_pattern = _roi_config(rois)
        download_dataset(preprocess_type="preprocessed", rois=rois)
        if uses_old_data:
            subject_dir = CACHE_PATH / "preprocessed" / f"sub-{subject:02d}"
        else:
            subject_dir = CACHE_PATH / "preprocessed_all" / f"sub-{subject:02d}__63_channels"
        x = np.load(subject_dir / f"preprocessed_eeg_{TYPE_DICT[data_type]}.npy", allow_pickle=True).item()
        x = _filter_channels(x, channel_pattern)
    else:
        download_dataset(preprocess_type="raw")
        x = run_preprocessing(
            subject, data_type, downsample_freq, l_freq, h_freq, tmin, tmax, baseline, tfr_n_bin, band_stop_n_bin, band_stop, rois,
            shuffle_reps=shuffle,
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

def load_events_list(subject, data_type, exclude_target):
    """Load the per-session event tables describing what was presented and when.

    Args:
    ----
        subject: subject number
        data_type: ``"train"`` or ``"test"``
        exclude_target: drop the target trials of the oddball task, keeping only the stimuli
            the participant was not responding to

    Returns:
    -------
        one event table per session

    """
    events_list = []
    for session in range(1, N_SESSIONS+1):
        raw_path = CACHE_PATH / "raw" / f"sub-{subject:02d}" / f"ses-{session:02d}" / f"raw_eeg_{TYPE_DICT[data_type]}.npy"
        
        eeg_data = np.load(raw_path, allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        del eeg_data
        
        events = mne.find_events(raw, stim_channel='stim', verbose=False)
        if exclude_target:
            idx_target = np.where(events[:,2] == 99999)[0]
            events = np.delete(events, idx_target, 0)
        
        events = events[:,2]
        events_list.append(events)
    return events_list
