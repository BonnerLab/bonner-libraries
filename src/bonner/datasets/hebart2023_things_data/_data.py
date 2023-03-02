from bonner.files import download_from_url
from bonner.datasets.hebart2023_things_data._utilities import CACHE_PATH

URLS = {
    "meg": "https://plus.figshare.com/ndownloader/files/37650461",
    "fmri": "https://plus.figshare.com/ndownloader/files/36806148",
    "brain_masks": "https://plus.figshare.com/ndownloader/files/36682242",
    "rois": "https://plus.figshare.com/ndownloader/files/38517326",
    "noise_ceilings": "https://plus.figshare.com/ndownloader/files/36682266",
    "flatmaps": "https://plus.figshare.com/ndownloader/files/36693528",
    "behavior": "https://plus.figshare.com/ndownloader/files/38787423",
}


if __name__ == "__main__":
    for data, url in URLS.items():
        download_from_url(url, filepath=CACHE_PATH / data, force=False)
