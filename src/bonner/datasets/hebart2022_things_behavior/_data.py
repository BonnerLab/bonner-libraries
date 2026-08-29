from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import os
import mne
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io
from tqdm.auto import tqdm
import itertools

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import unzip
from osfclient.api import OSF

IDENTIFIER = "hebart2022.things.behavior"
PROJECT_ID = "f5rn6"

CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def _download_osf_project(project_id, save_path, target_paths=None, use_cached=True):
    if (not use_cached) or (not save_path.exists()):
        osf = OSF()
        project = osf.project(project_id)
        storage = project.storage('osfstorage')
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

def _download_all():
    _download_osf_project(
        project_id=PROJECT_ID,
        save_path=CACHE_PATH,
        target_paths=(
            "/data/spose_embedding_66d_sorted.txt",
            "/data/spose_similarity.mat",
            "/data/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv",
            "/variables/labels.txt",
            "/variables/unique_id.txt",
        ),
    )

def load_embeddings(scale: bool = False):
    """Load the SPoSE behavioural embedding of the THINGS object concepts.

    Each object is described by non-negative, sparsely-loading dimensions derived from human
    triplet-odd-one-out judgements, which is why the dimensions are labelled and interpretable
    rather than arbitrary.

    Downloads the source files on first use.

    Args:
    ----
        scale: standardize each dimension. Currently raises — the implementation calls numpy's
            standard deviation with a torch keyword — so leave it at its default

    Returns:
    -------
        the embedding, dimensions ``("object", "behavior")``

    """
    _download_all()
    
    embd = pd.read_csv(CACHE_PATH / "data" / "spose_embedding_66d_sorted.txt", sep="\t", header=None).values
    bhv = pd.read_csv(CACHE_PATH / "variables" / "labels.txt", sep="\t", header=None).values.flatten()
    object = pd.read_csv(CACHE_PATH / "variables" / "unique_id.txt", sep="\t", header=None).values.flatten()
    
    if scale:
        embd /= embd.std(dim=0)
    
    return xr.DataArray(
        embd,
        dims=("object", "behavior"),
        coords={"object": object, "behavior": bhv},
    )
    
def load_object_labels():
    """Load the unique object identifiers, in the row order the other loaders use.

    Returns:
    -------
        object identifiers

    """
    _download_all()
    
    return  pd.read_csv(CACHE_PATH / "variables" / "unique_id.txt", sep="\t", header=None).values.flatten()
    
def load_spose_rsm():
    """Load the object-by-object similarity matrix predicted by the SPoSE embedding.

    This is the model's reconstruction of the behavioural similarities, not the raw judgements;
    the triplet responses those were fitted to are returned by ``load_triplet_results``.

    Returns:
    -------
        the similarity matrix, dimensions ``("object0", "object1")``

    """
    _download_all()
    
    spose_similarity = scipy.io.loadmat(CACHE_PATH / "data" / "spose_similarity.mat")["spose_sim"]
    object = load_object_labels()
    return xr.DataArray(
        spose_similarity,
        dims=("object0", "object1"),
        coords={"object0": object, "object1": object},
    )
    
def load_triplet_results():
    """Load the raw odd-one-out triplet judgements the embedding was fitted to.

    Returns:
    -------
        one row per triplet judgement

    """
    _download_all()
    
    return pd.read_csv(CACHE_PATH / "data" / "triplet_dataset" /"triplets_large_final_correctednc_correctedorder.csv", sep="\t")
    
    
    
