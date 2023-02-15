import os
from pathlib import Path


BONNER_MODELS_CACHE = Path(
    os.getenv("BONNER_MODELS_CACHE", str(Path.home() / ".cache" / "bonner-models"))
)
