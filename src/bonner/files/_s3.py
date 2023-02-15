from pathlib import Path

from loguru import logger
import boto3

# TODO: refactor to include bonner.files._utilities.prepare_filepath


def download(
    s3_path: Path,
    *,
    bucket: str,
    local_path: Path = None,
    use_cached: bool = True,
) -> None:
    """Download file(s) from S3.

    Args:
        s3_path: path of file in S3
        bucket: S3 bucket name
        local_path: local path of file
        use_cached: use existing file or re-download, defaults to True
    """
    if local_path is None:
        local_path = s3_path
    s3 = boto3.client("s3")
    if (not use_cached) or (not local_path.exists()):
        logger.debug(f"Downloading {s3_path} from S3 bucket {bucket} to {local_path}")
        local_path.parent.mkdir(exist_ok=True, parents=True)
        with open(local_path, "wb") as f:
            s3.download_fileobj(bucket, str(s3_path), f)
    else:
        logger.debug(
            "Using previously downloaded file at"
            f" {local_path} instead of downloading {s3_path} from S3 bucket"
            f" {bucket}"
        )
