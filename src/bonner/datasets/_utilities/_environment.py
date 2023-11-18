import os
from pathlib import Path

BONNER_DATASETS_HOME = Path(
    os.getenv("BONNER_DATASETS_HOME", str(Path.home() / ".cache" / "bonner-datasets")),
)
