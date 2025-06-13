from collections.abc import Collection
from pathlib import Path

from osfclient.api import OSF


def download(
    *,
    project_id: str,
    directory: Path,
    storage: str = "osfstorage",
    files: Collection[str] | None = None,
    use_cached: bool = True,
) -> None:
    osf = OSF()
    project = osf.project(project_id)

    files_to_download = None if files is None else set(files)

    if use_cached:
        files_to_download = {
            file_
            for file_ in files
            if not (directory / Path(file_).relative_to("/")).exists()
        }
        if len(files_to_download) == 0:
            return

    if (not use_cached) or (not directory.exists()):
        directory.mkdir(exist_ok=True, parents=True)
        for file_ in project.storage(storage).files:
            if (files_to_download is None) or (file_.path in files_to_download):
                filepath = directory / Path(file_.path).relative_to("/")
                filepath.parent.mkdir(exist_ok=True, parents=True)
                with filepath.open("wb") as f:
                    file_.write_to(f)

                if files_to_download is not None:
                    files_to_download.remove(file_.path)

            if (files_to_download is not None) and len(files_to_download) == 0:
                break
