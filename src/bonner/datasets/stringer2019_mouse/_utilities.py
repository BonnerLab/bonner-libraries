import xarray as xr
from scipy.sparse.linalg import eigsh

from bonner.datasets._utilities import BONNER_DATASETS_HOME

IDENTIFIER = "stringer2019.mouse"
SESSIONS = (
    {
        "mouse": "M160825_MP027",
        "date": "2016-12-14",
    },
    {
        "mouse": "M161025_MP030",
        "date": "2017-05-29",
    },
    {
        "mouse": "M170604_MP031",
        "date": "2017-06-28",
    },
    {
        "mouse": "M170714_MP032",
        "date": "2017-08-07",
    },
    {
        "mouse": "M170714_MP032",
        "date": "2017-09-14",
    },
    {
        "mouse": "M170717_MP033",
        "date": "2017-08-20",
    },
    {
        "mouse": "M170717_MP034",
        "date": "2017-09-11",
    },
)
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def preprocess_assembly(assembly: xr.Dataset, *, denoise: bool = True) -> xr.DataArray:
    if denoise:
        spontaneous = assembly["spontaneous activity"]
        mean = spontaneous.mean("time")
        std = spontaneous.std("time") + 1e-6

        spontaneous = (spontaneous - mean) / std

        stimulus_related = (assembly["stimulus-related activity"] - mean) / std

        stimulus_related = stimulus_related.isel(
            {"presentation": stimulus_related["stimulus"].values != 2800}
        )

        _, eigenvectors = eigsh(spontaneous.values.T @ spontaneous.values, k=32)
        stimulus_related -= (stimulus_related.values @ eigenvectors) @ eigenvectors.T
        stimulus_related -= stimulus_related.mean("presentation")
    else:
        stimulus_related = assembly["stimulus-related activity"]
        stimulus_related = stimulus_related.isel(
            {"presentation": stimulus_related["stimulus"].values != 2800}
        )
    return (
        stimulus_related
        .transpose("presentation", "neuroid")
        .sortby(["x", "y", "z"])
        .sortby(["stimulus", "repetition"])
        .assign_attrs(assembly.attrs)
    )