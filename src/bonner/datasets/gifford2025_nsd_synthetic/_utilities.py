__all__ = (
    "compute_shared_stimuli",
    "create_roi_selector",
)

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.datasets.allen2021_natural_scenes._utilities import (
    compute_shared_stimuli,
    create_roi_selector,
)

IDENTIFIER = "gifford2025.nsd_synthetic"
BUCKET_NAME = "natural-scenes-dataset"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER
N_SUBJECTS = 8
