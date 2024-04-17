from pathlib import Path

from bonner.datasets._utilities import BONNER_DATASETS_HOME
from bonner.files import download_from_s3

IDENTIFIER = "iarpa.microns"
CACHE_PATH = BONNER_DATASETS_HOME / IDENTIFIER


def download() -> None:
    filename = "functional_data_database_container_image_v8.tar"
    download_from_s3(
        Path(
            "iarpa_microns/minnie/functional_data/two_photon_processed_data_and_metadata/database_v8/functional_data_database_container_image_v8.tar",
        ),
        local_path=CACHE_PATH / "downloads" / filename,
        bucket="bossdb-open-data",
    )


if __name__ == "__main__":
    download()
