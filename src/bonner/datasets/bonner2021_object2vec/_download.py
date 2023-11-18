from pathlib import Path

from bonner.datasets.bonner2021_object2vec._utilities import FILENAMES, N_SUBJECTS, URLS
from bonner.files import download_from_url, unzip


def download_dataset(*, force: bool) -> None:
    filepath = download_from_url(
        URLS["stimuli"],
        filepath=Path(FILENAMES["stimuli"]),
        force=force,
    )
    unzip(filepath, extract_dir=Path.cwd(), remove_zip=False)

    download_from_url(
        URLS["conditions"],
        filepath=Path(FILENAMES["conditions"]),
        force=force,
    )

    for subject in range(N_SUBJECTS):
        for filetype in ("activations", "noise_ceilings", "rois", "cv_sets"):
            download_from_url(
                URLS[filetype][subject],
                filepath=Path(FILENAMES[filetype][subject]),
                force=force,
            )
        for urls, filenames in zip(
            URLS["contrasts"].values(),
            FILENAMES["contrasts"].values(),
            strict=True,
        ):
            download_from_url(
                urls[subject],
                filepath=Path(filenames[subject]),
                force=force,
            )
