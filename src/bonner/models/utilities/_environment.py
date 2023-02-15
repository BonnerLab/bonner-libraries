import os
from pathlib import Path


BONNER_MODELS_HOME = Path(
    os.getenv("BONNER_MODELS_HOME", str(Path.home() / ".cache" / "bonner-models"))
)
