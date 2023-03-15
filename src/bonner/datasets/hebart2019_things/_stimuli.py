from bonner.files import download_from_url, unzip
from bonner.datasets.hebart2019_things._utilities import CACHE_PATH, PASSWORD

URL = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip="


def download_stimuli(force: bool = False) -> None:
    path = CACHE_PATH / "download" / "things.zip"
    download_from_url(URL, filepath=path, force=force)

    unzip(path, extract_dir=CACHE_PATH, password=PASSWORD, remove_zip=False)
    path = CACHE_PATH / "THINGS" / "Images"
    for suffix in ("A-C", "D-K", "L-Q", "R-S", "T-Z"):
        unzip(
            path / f"object_images_{suffix}.zip",
            extract_dir=CACHE_PATH / "images",
            remove_zip=False,
            password=PASSWORD,
        )


if __name__ == "__main__":
    download_stimuli()
