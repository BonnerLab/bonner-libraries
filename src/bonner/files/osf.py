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

    files_remaining = None if files is None else set(files)

    if (not use_cached) or (not directory.exists()):
        directory.mkdir(exist_ok=True, parents=True)
        for file_ in project.storage(storage).files:
            if (files_remaining is None) or (file_.path in files_remaining):
                filepath = directory / Path(file_.path).relative_to("/")
                filepath.parent.mkdir(exist_ok=True, parents=True)
                with filepath.open("wb") as f:
                    file_.write_to(f)

                if files_remaining is not None:
                    files_remaining.remove(file_.path)

            if (files_remaining is not None) and len(files_remaining) == 0:
                break
