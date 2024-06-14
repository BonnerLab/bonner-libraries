from pathlib import Path

import boto3
from loguru import logger

# TODO: refactor to include bonner.files._utilities.prepare_filepath


def download(
    s3_path: Path,
    *,
    bucket: str,
    local_path: Path | None = None,
    use_cached: bool = True,
    is_dir: bool = False,
) -> None:
    """Download file(s) from S3.

    Args:
    ----
        s3_path: path of file or directory in S3
        bucket: S3 bucket name
        local_path: local path of file or directory
        use_cached: use existing file or re-download, defaults to True
        dir: download all files in the directory if True, defaults to False
    """
    s3 = boto3.client("s3")
    
    if dir:
        # Download all files in the directory
        if local_path is None:
            local_path = Path(s3_path)
        
        paginator = s3.get_paginator("list_objects_v2")
        response_iterator = paginator.paginate(Bucket=bucket, Prefix=str(s3_path))

        for page in response_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_file_path = obj["Key"]
                    local_file_path = local_path / Path(s3_file_path).relative_to(s3_path)
                    
                    if (not use_cached) or (not local_file_path.exists()):
                        logger.debug(f"Downloading {s3_file_path} from S3 bucket {bucket} to {local_file_path}")
                        local_file_path.parent.mkdir(exist_ok=True, parents=True)
                        with local_file_path.open("wb") as f:
                            s3.download_fileobj(bucket, s3_file_path, f)
                    else:
                        logger.debug(
                            "Using previously downloaded file at"
                            f" {local_file_path} instead of downloading {s3_file_path} from S3 bucket"
                            f" {bucket}",
                        )
    else:
        # Download a single file
        if local_path is None:
            local_path = s3_path
        if (not use_cached) or (not local_path.exists()):
            logger.debug(f"Downloading {s3_path} from S3 bucket {bucket} to {local_path}")
            local_path.parent.mkdir(exist_ok=True, parents=True)
            with local_path.open("wb") as f:
                s3.download_fileobj(bucket, str(s3_path), f)
        else:
            logger.debug(
                "Using previously downloaded file at"
                f" {local_path} instead of downloading {s3_path} from S3 bucket"
                f" {bucket}",
            )

