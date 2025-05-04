import pandas as pd
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import osf

IDENTIFIER = "hebart2023.things-data"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER / "behavior"


def load_embeddings() -> xr.DataArray:
    osf.download(
        project_id="f5rn6",
        directory=CACHE_PATH,
        files=(
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
