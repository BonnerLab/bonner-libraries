from typing import Any
from pathlib import Path
import requests

from bonner.files._utilities import prepare_filepath


def download(
    file_id: str, *, filepath: Path, chunk_size: int = 32768, force: bool = True
) -> Path:
    url = "https://docs.google.com/uc?export=download"

    filepath = prepare_filepath(
        filepath=filepath,
        url=f"{url}&{file_id}",
        force=force,
    )
    if filepath.exists():
        return filepath

    session = requests.Session()

    response = session.get(url, params={"id": file_id, "confirm": 1}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    return filepath


def _get_confirm_token(response: requests.Response) -> Any:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None
