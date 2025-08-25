from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import xarray as xr

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import osf

IDENTIFIER = "hebart2023.things-data"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER / "behavior"


def load_embeddings() -> xr.DataArray:
    osf.download(
        project_id="f5rn6",
        directory=CACHE_PATH,
        files={
            "/data/spose_embedding_66d_sorted.txt",
            "/variables/labels.txt",
            "/variables/unique_id.txt",
        },
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


def load_triplets() -> pd.DataFrame:
    directory = Path("data/triplet_dataset")

    filenames = (
        "trainset.txt",
        "testset1.txt",
        "testset2.txt",
        "testset2_repeat.txt",
        "testset3.txt",
        "validationset.txt",
        "triplets_large_final_correctednc_correctedorder.csv",
    )
    osf.download(
        project_id="f5rn6",
        directory=CACHE_PATH,
        files={f"/{directory / filename}" for filename in filenames},
    )
    triplets = pd.read_csv(
        CACHE_PATH / directory / "triplets_large_final_correctednc_correctedorder.csv",
        sep="\t",
        header=0,
        dtype={
            "image1": np.uint64,
            "image2": np.uint64,
            "image3": np.uint64,
            "choice": np.uint8,
            "RT": np.float32,
            "noise_ceiling": np.float64,
            "subject_id": pd.CategoricalDtype(),
            "HIT_nr": np.uint64,
            "trial_nr": np.uint64,
            "age": np.float64,
            "gender": pd.CategoricalDtype(),
            "dataset": np.uint64,
        },
    )

    triplets["datetime"] = pd.to_datetime(triplets["date"] + " " + triplets["time"])
    triplets = triplets.drop(columns=["date", "time"])

    for idx in range(3):
        # zero-index all indices
        triplets[f"image{idx + 1}"] -= 1

    triplets["choice"] = triplets.iloc[:, :3].to_numpy()[
        np.arange(len(triplets)),
        triplets["choice"].to_numpy() - 1,
    ]

    # make first two columns the foil trials, and the third column the odd-one-out
    triplets.iloc[..., :3] = standardize_order_of_triplets(
        triplets.iloc[..., :3].to_numpy(),
        choices=triplets["choice"].to_numpy(),
    )
    triplets = triplets.rename(columns={"image1": "image_1", "image2": "image_2"}).drop(
        columns="image3",
    )

    # tuple for each triplet to use for indexing (since it can be a hashable unique ID)
    triplets.insert(
        0,
        "triplet",
        [
            tuple(x)
            for x in np.sort(triplets.iloc[:, :3].to_numpy().copy(), axis=-1).tolist()
        ],
    )

    # add metadata about whether each triplet is in the train, test, or validation sets
    for subset in (
        "trainset",
        "testset1",
        "testset2",
        "testset2_repeat",
        "testset3",
        "validationset",
    ):
        filepath = CACHE_PATH / directory / f"{subset}.txt"
        array = np.loadtxt(filepath).astype(np.uint64)
        triplets = triplets.assign(
            **{
                subset: triplets["triplet"].isin(
                    {tuple(x) for x in np.sort(array, axis=-1).tolist()},
                ),
            },
        )

    return triplets


def convert_indices_to_labels(
    indices: npt.NDArray[np.uint64],
    *,
    embeddings: xr.DataArray,
) -> npt.NDArray:
    return embeddings["object"].to_numpy()[indices]


def load_densely_sampled_triplet_categories() -> set[str]:
    directory = Path("variables")

    filenames = ("words48.mat",)
    osf.download(
        project_id="f5rn6",
        directory=CACHE_PATH,
        files={f"/{directory / filename}" for filename in filenames},
    )
    objects = {
        object_.replace(" ", "_")
        for object_ in scipy.io.loadmat(
            CACHE_PATH / directory / filenames[0],
            simplify_cells=True,
        )["words48"]
    }
    return (
        objects  # FIXME(add camera1/camera2 or file1/file2 instead of camera and file)
    )
    # objects.remove("camera")


def standardize_order_of_triplets(
    triplets: npt.NDArray[np.uint64],
    *,
    choices: npt.NDArray[np.uint64],
) -> npt.NDArray[np.uint64]:
    """
    Standardize the order of triplets for the odd-one-out task.

    Given a triplet of stimulus indices `{j, k, i}`, sort it to `(i, j, k)` such
    that `k` is the chosen odd-stimulus-out and `i < j`.

    Parameters
    ----------
    triplets
        `(n_triplets, 3)` matrix of stimulus indices, where each row
            represents a triplet `(i, j, k)` of stimuli,
    choices
        `(n_triplets,)` vector of stimulus choices, where each element is an
        index in `{i, j, k}`.

    Returns
    -------
        `(n_triplets, 3)` matrix of stimulus indices, sorted such that `k` is the
        chosen odd-stimulus-out and `i < j`.

    """
    return np.concatenate(
        [
            triplets[
                triplets
                != np.tile(
                    choices,
                    reps=(3, 1),
                ).T,
            ].reshape((len(triplets), 2)),
            choices.reshape(-1, 1),
        ],
        axis=-1,
    )
