from pathlib import Path

import pandas as pd


def create_stimulus_set() -> pd.DataFrame:
    parent_dir = Path("BOLD5000_Stimuli") / "Scene_Stimuli" / "Presented_Stimuli"
    image_paths = list(parent_dir.rglob("*.*"))
    return pd.DataFrame(
        {
            "stimulus": [path.stem for path in image_paths],
            "dataset": [path.parent.name for path in image_paths],
            "filename": [str(path.relative_to(parent_dir)) for path in image_paths],
        },
    )
