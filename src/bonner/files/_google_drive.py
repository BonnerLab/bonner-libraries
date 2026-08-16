from pathlib import Path
from typing import Any

import requests
from bonner.files._utilities import prepare_filepath


#: ⚠ The OLD endpoint (``docs.google.com/uc?export=download`` plus a ``download_warning`` cookie) is
#: dead: measured 2026-08-16 it answers 303 -> 500 for a large file. Google's current large-file path
#: is this host, and ``confirm=t`` clears the interstitial in one hop -- verified against a 2.5 GB
#: file and against a 94 MB one whose bytes match an existing local copy exactly.
_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def download(
    file_id: str,
    *,
    filepath: Path,
    chunk_size: int = 32_768,
    force: bool = True,
) -> Path:
    existed = filepath.exists()
    filepath = prepare_filepath(filepath=filepath, force=force)
    if existed and not force:
        return filepath

    session = requests.Session()

    params = {"id": file_id, "export": "download", "confirm": "t"}
    response = session.get(_DOWNLOAD_URL, params=params, stream=True)
    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(_DOWNLOAD_URL, params=params, stream=True)
    response.raise_for_status()

    # ⚠⚠ The characteristic Drive failure is a 200 carrying the interstitial HTML, which lands on
    # disk as a plausible-looking file and only fails much later at `torch.load`. Fail here instead.
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        msg = (
            f"Google Drive returned HTML, not a file, for id={file_id!r} "
            f"(content-type={content_type!r}). The file is probably permission-gated or the "
            "confirm flow has changed again."
        )
        raise RuntimeError(msg)

    with filepath.open("wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    return filepath


def _get_confirm_token(response: requests.Response) -> Any:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None
