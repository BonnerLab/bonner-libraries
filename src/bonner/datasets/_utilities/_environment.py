from pathlib import Path
import os

BONNER_DATASETS_CACHE = Path(
    os.getenv("BONNER_DATASETS_CACHE", str(Path.home() / ".cache" / "bonner-datasets"))
)
