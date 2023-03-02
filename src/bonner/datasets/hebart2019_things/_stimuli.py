from bonner.files import download_from_url, unzip
from bonner.datasets.hebart2019_things._utilities import CACHE_PATH, PASSWORD

URL = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/?zip="


def download_stimuli(force: bool = False):
    path = CACHE_PATH / "things.zip"
    download_from_url(URL, filepath=CACHE_PATH / "things.zip", force=force)

    unzip(path, extract_dir=CACHE_PATH, password=PASSWORD)
