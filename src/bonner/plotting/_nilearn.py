import nilearn.plotting
import numpy as np
from matplotlib.axes import Axes
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data, load_surf_mesh, vol_to_surf


def _normalize_curv_map(
    curv_map,
    /,
    *,
    low: float = 0.25,
    high: float = 0.5,
) -> np.ndarray:
    negative = curv_map < 0
    positive = curv_map >= 0
    curv_map[negative] = low
    curv_map[positive] = high
    return curv_map


def plot_brain_map(
    volume: Nifti1Image,
    *,
    ax: Axes,
    hemisphere: str,
    surface_type: str = "infl",
    mesh: str = "fsaverage",
    low: float = 0.25,
    high: float = 0.5,
    **kwargs,
) -> None:
    fsaverage = fetch_surf_fsaverage(mesh=mesh)

    nilearn.plotting.plot_surf_stat_map(
        axes=ax,
        stat_map=vol_to_surf(
            volume,
            fsaverage[f"pial_{hemisphere}"],
        ),
        surf_mesh=load_surf_mesh(fsaverage[f"{surface_type}_{hemisphere}"]),
        threshold=np.finfo(np.float32).resolution,
        colorbar=False,
        bg_map=_normalize_curv_map(
            load_surf_data(fsaverage[f"curv_{hemisphere}"]),
            low=low,
            high=high,
        ),
        engine="matplotlib",
        **kwargs,
    )
