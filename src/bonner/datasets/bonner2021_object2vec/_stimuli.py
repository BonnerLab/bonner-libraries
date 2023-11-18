from pathlib import Path

import numpy as np
import pandas as pd
from bonner.datasets.bonner2021_object2vec._utilities import (
    FILENAMES,
    N_SUBJECTS,
    load_conditions,
)
from scipy.io import loadmat


def create_stimulus_set() -> pd.DataFrame:
    conditions = load_conditions()
    metadata = {}
    metadata["condition"] = conditions
    for subject in range(N_SUBJECTS):
        sets = loadmat(FILENAMES["cv_sets"][subject], simplify_cells=True)["sets"]
        cv_set = np.empty(len(conditions), dtype=np.int8)
        for i_set, set_ in enumerate(sets):
            cv_set[np.isin(conditions, set_)] = i_set
        metadata[f"cv_set_subject{subject}"] = cv_set
    metadata = pd.DataFrame.from_dict(metadata)

    paths = [
        path
        for path in sorted(Path("stimuli").rglob("*.*"))
        if path.suffix in (".jpg", ".png")
    ]
    stimulus_set = pd.DataFrame.from_dict(
        {
            "stimulus": [
                f"{path.stem}{path.parent.name.split('_')[0]}" for path in paths
            ],
            "filename": [str(path) for path in paths],
            "condition": [path.stem[:-3] for path in paths],
            "background": [path.parent.name.split("_")[0] for path in paths],
        },
    )
    return stimulus_set.merge(metadata, on="condition")
