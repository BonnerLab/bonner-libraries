from pathlib import Path
import requests

from loguru import logger

from bonner.files._utilities import prepare_filepath


def download(
    url: str,
    *,
    filepath: Path = None,
    stream: bool = True,
    allow_redirects: bool = True,
    chunk_size: int = 1024**2,
    force: bool = True,
) -> Path:
    filepath = prepare_filepath(
        filepath=filepath,
        url=url,
        force=force,
    )
    if filepath.exists():
        return filepath

    logger.debug(f"Downloading from {url} to {filepath}")
    r = requests.Session().get(url, stream=stream, allow_redirects=allow_redirects)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)

    return filepath
