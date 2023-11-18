import subprocess
from pathlib import Path

from bonner.datasets.chang2019_bold5000._utilities import (
    FIGSHARE_ARTICLE_ID_V1,
    FIGSHARE_ARTICLE_ID_V2,
    N_SESSIONS,
    N_SUBJECTS,
    S3_ROI_MASKS,
    URL_IMAGES,
    get_betas_filename,
    get_brain_mask_filename,
    get_imagenames_filename,
)
from bonner.files import download_from_url, unzip
from bonner.files._figshare import get_url_dict
from tqdm import tqdm


def download_dataset(*, force: bool = False, **kwargs: str) -> None:
    urls = get_url_dict(FIGSHARE_ARTICLE_ID_V2)

    for subject in tqdm(range(N_SUBJECTS), desc="subject", leave=False):
        filenames = [
            get_brain_mask_filename(subject),  # brain masks
            get_imagenames_filename(subject),  # image names
        ]
        for filename in filenames:
            download_from_url(urls[filename], Path(filename), force=force)
        for session in tqdm(range(N_SESSIONS[subject]), desc="session", leave=False):
            filename = get_betas_filename(subject, session)  # betas
            download_from_url(urls[filename], Path(filename), force=force)

    urls = get_url_dict(FIGSHARE_ARTICLE_ID_V1)
    urls = {
        "BOLD5000_Structural.zip": urls["BOLD5000_Structural.zip"],
        "stimuli.zip": URL_IMAGES,
    }
    for filename, url in urls.items():
        filepath = download_from_url(url, filepath=Path(filename), force=force)
        unzip(filepath, extract_dir=Path.cwd(), remove_zip=False)

    # ROI masks
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            f"{S3_ROI_MASKS}",
            f"{Path.cwd()}",
        ],
    )
    # rename some inconsistently named files
    remapping = {
        "sub-CSI2/sub-CSI2_mask-LHLO.nii.gz": "sub-CSI2/sub-CSI2_mask-LHLOC.nii.gz",
        "sub-CSI2/sub-CSI2_mask-RHLO.nii.gz": "sub-CSI2/sub-CSI2_mask-RHLOC.nii.gz",
        "sub-CSI2/sub-CSI2_mask-RHRRSC.nii.gz": "sub-CSI2/sub-CSI2_mask-RHRSC.nii.gz",
        "sub-CSI3/sub-CSI3_mask-LHLO.nii.gz": "sub-CSI3/sub-CSI3_mask-LHLOC.nii.gz",
        "sub-CSI3/sub-CSI3_mask-RHLO.nii.gz": "sub-CSI3/sub-CSI3_mask-RHLOC.nii.gz",
    }
    for filename_old, filename_new in remapping.items():
        Path(filename_old).replace(Path(filename_new))
