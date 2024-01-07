import uuid
from pathlib import Path

from loguru import logger


def prepare_filepath(*, force: bool, filepath: Path | None) -> Path:
    # TODO(Raj): replace with tempfile.NamedTemporaryFile
    if filepath is None:
        filepath = Path("/tmp") / f"{uuid.uuid4()}"
    elif filepath.exists():
        if not force:
            logger.debug(f"Using existing file at {filepath}")
        else:
            filepath.unlink()
    filepath.parent.mkdir(exist_ok=True, parents=True)
    return filepath
