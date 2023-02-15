from pathlib import Path
import os

BONNER_DATASETS_HOME = Path(
    os.getenv("BONNER_DATASETS_HOME", str(Path.home() / ".cache" / "bonner-datasets"))
)
