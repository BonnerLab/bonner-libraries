from pathlib import Path
import uuid

from loguru import logger


def prepare_filepath(*, force: bool, filepath: Path, url: str) -> Path:
    if filepath is None:
        filepath = Path("/tmp") / f"{uuid.uuid4()}"
    elif filepath.exists():
        if not force:
            logger.debug(
                "Using previously downloaded file at"
                f" {filepath} instead of downloading from {url}"
            )
        else:
            filepath.unlink()
    filepath.parent.mkdir(exist_ok=True, parents=True)
    return filepath
