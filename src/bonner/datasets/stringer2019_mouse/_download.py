from pathlib import Path

from bonner.files._figshare import get_url_dict
from bonner.files import download_from_url

FIGSHARE_ARTICLE_ID = 6845348


def download_dataset(force: bool = False) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID)
    urls_data = {key: url for key, url in urls.items() if "natimg2800_M" in key}
    for filename, url in urls_data.items():
        download_from_url(url, filepath=Path(filename), force=force)
    filename = "images_natimg2800_all.mat"
    download_from_url(urls[filename], filepath=Path(filename), force=force)
