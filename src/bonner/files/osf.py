from pathlib import Path

from osfclient.api import OSF


def download(
    *,
    project_id: str,
    directory: Path,
    storage: str = "osfstorage",
    files: set[str] | None = None,
    use_cached: bool = True,
) -> None:
    directory.mkdir(exist_ok=True, parents=True)

    # short-circuit network call if required files already exist
    if (
        use_cached
        and files is not None
        and all((directory / Path(file_).relative_to("/")).exists() for file_ in files)
    ):
        return

    project = OSF().project(project_id)

    for file_ in project.storage(storage).files:
        if files is None or file_.path in files:
            filepath = directory / Path(file_.path).relative_to("/")
            filepath.parent.mkdir(exist_ok=True, parents=True)

            if filepath.exists() and use_cached:
                continue

            with filepath.open("wb") as f:
                file_.write_to(f)

            if files is not None:
                files.remove(file_.path)

                if len(files) == 0:
                    return
