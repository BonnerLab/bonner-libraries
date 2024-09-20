import logging
import os
from pathlib import Path

import pandas as pd
import xarray as xr
from osfclient.api import OSF

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import unzip

logging.basicConfig(level=logging.INFO)


IDENTIFIER = "hebart2022.things.behavior"
PROJECT_ID = "f5rn6"

CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def _download_osf_project(
    project_id,
    save_path,
    target_paths=None,
    use_cached: bool = True,
) -> None:
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage("osfstorage")

    if (not use_cached) or (not save_path.exists()):
        os.makedirs(save_path, exist_ok=True)
        for file in storage.files:
            if target_paths is not None and file.path not in target_paths:
                continue
            file_path = os.path.join(save_path, file.path.lstrip("/"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as local_file:
                file.write_to(local_file)

            if file_path.endswith(".zip"):
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

    embeddings = pd.read_csv(
        CACHE_PATH / "data" / "spose_embedding_66d_sorted.txt",
        sep="\t",
        header=None,
    ).to_numpy()
    behavior = (
        pd.read_csv(
            CACHE_PATH / "variables" / "labels.txt",
            sep="\t",
            header=None,
        )
        .to_numpy()
        .flatten()
    )
    object_ids = (
        pd.read_csv(
            CACHE_PATH / "variables" / "unique_id.txt",
            sep="\t",
            header=None,
        )
        .to_numpy()
        .flatten()
    )

    return xr.DataArray(
        embeddings,
        dims=("object", "behavior"),
        coords={"object": object_ids, "behavior": behavior},
    )
