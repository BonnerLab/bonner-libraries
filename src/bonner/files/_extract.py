from pathlib import Path
import tarfile
import zipfile

from loguru import logger


def untar(filepath: Path, *, extract_dir: Path = None, remove_tar: bool = True) -> Path:
    if extract_dir is None:
        extract_dir = Path("/tmp")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(filepath) as tar:
        if all([(extract_dir / filename).exists() for filename in tar.getnames()]):
            logger.debug(
                f"Using previously extracted files from {extract_dir}"
                f" instead of extracting from {filepath}"
            )
        else:
            logger.debug(f"Extracting from {filepath} to {extract_dir}")
            tar.extractall(path=extract_dir)

    if remove_tar:
        logger.debug(f"Deleting {filepath} after extraction")
        filepath.unlink()

    return extract_dir


def unzip(filepath: Path, *, extract_dir: Path = None, remove_zip: bool = True) -> Path:
    if extract_dir is None:
        extract_dir = Path("/tmp")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(filepath, "r") as f:
        if all([(extract_dir / filename).exists() for filename in f.namelist()]):
            logger.debug(
                f"Using previously extracted files from {extract_dir}"
                f" instead of extracting from {filepath}"
            )
        else:
            logger.debug(f"Extracting from {filepath} to {extract_dir}")
            f.extractall(extract_dir)

    if remove_zip:
        logger.debug(f"Deleting {filepath} after extraction")
        filepath.unlink()

    return extract_dir
