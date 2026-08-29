from pathlib import Path
from typing import Any

import requests
from bonner.files._utilities import prepare_filepath


#: Google Drive's large-file download host. The older ``docs.google.com/uc?export=download``
#: endpoint, paired with a ``download_warning`` cookie, no longer serves large files — it redirects
#: and then fails — so it must not be reinstated. Passing ``confirm=t`` to this host clears the
#: virus-scan interstitial in a single request.
_DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def download(
    file_id: str,
    *,
    filepath: Path,
    chunk_size: int = 32_768,
    force: bool = True,
) -> Path:
    """Download a file from Google Drive by its file id.

    Streams to disk in chunks, so the file need not fit in memory. Drive gates large files behind
    a virus-scan interstitial; both the direct confirmation and the cookie-based one are handled.

    Args:
    ----
        file_id: the Drive file id
        filepath: where to write the file
        chunk_size: bytes per streamed chunk
        force: re-download and overwrite even when the destination already exists

    Returns:
    -------
        the path written to

    Raises:
    ------
        RuntimeError: if Drive serves the interstitial page instead of the file, which usually
            means the file is permission-gated

    """
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

    # Drive signals refusal with a 200 carrying the interstitial HTML, so the status code alone
    # cannot detect it: without this check the page is written out as a plausible-looking file and
    # the failure surfaces much later, wherever that file is first parsed.
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
