import hashlib
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr

BONNER_BRAINIO_HOME = Path(
    os.getenv("BONNER_BRAINIO_HOME", str(Path.home() / ".cache" / "bonner-brainio")),
)


def validate_catalog(path: Path) -> None:
    """Validate a BrainIO Catalog.

    Ensures that the Catalog complies with the BrainIO specification*.

    * Warning: This does NOT check whether the Data Assemblies and Stimulus Sets stored in the Catalog are themselves valid.
    * Warning: This does NOT check whether the 'identifier' and 'stimulus_set_identifier' columns of a row corresponding to a Data Assembly netCDF-4 file match its global attributes of the same name.
    * Warning: This does NOT check whether the all the entries in the 'stimulus_set_identifier' column are present a

    Args:
    ----
        path: path to the Catalog CSV file

    """
    column_headers_are_unique = pd.read_csv(
        path,
        nrows=1,
        dtype=str,
    ).columns.is_unique
    if not column_headers_are_unique:
        error = f"The column headers of the Catalog CSV file {path} MUST be unique"
        raise ValueError(error)

    catalog = pd.read_csv(path, dtype=str)
    required_columns = {
        "sha1",
        "lookup_type",
        "identifier",
        "stimulus_set_identifier",
        "location_type",
        "location",
        "class",
    }
    for required_column in required_columns:
        if required_column not in catalog.columns:
            error = (
                f"'{required_column}' MUST be a column of the Catalog CSV file {path}"
            )
            raise ValueError(error)

    for column in catalog.columns:
        if not re.match(r"^[a-z0-9_]+$", column):
            error = (
                f"The column header '{column}' of the Catalog CSV file {path} MUST contain"
                " only lowercase alphabets, digits, and underscores"
            )
            raise ValueError(error)

    if catalog.empty:
        return

    for column in ("identifier", "stimulus_set_identifier"):
        if catalog.dtypes[column] != "O":
            error = (
                f"The column '{column}' of the Catalog CSV file {path} MUST have string"
                " entries"
            )
            raise ValueError(error)

    for sha1 in catalog["sha1"]:
        if not (re.match(r"^[a-fA-F0-9]+$", sha1) and len(sha1) == 40):
            error = f"The SHA1 hash {sha1} in the Catalog CSV file {path} is invalid"
            raise ValueError(error)

    if not (
        catalog.loc[catalog["lookup_type"] == "assembly"]
        .groupby("identifier")
        .count()["sha1"]
        == 1
    ).all():
        error = (
            "Each Data Assembly MUST have exactly 1 corresponding row in the Catalog CSV"
            f" file {path}"
        )
        raise ValueError(error)

    if not (
        catalog.loc[catalog["lookup_type"] == "stimulus_set"]
        .groupby("identifier")
        .count()["sha1"]
        == 2
    ).all():
        error = (
            "Each Stimulus Set MUST have exactly 2 corresponding rows in the Catalog CSV"
            f" file {path}"
        )
        raise ValueError(error)

    if not catalog["sha1"].is_unique:
        error = f"The 'sha1' column of the Catalog CSV file {path} MUST contain unique entries"
        raise ValueError(error)

    if not set(catalog["lookup_type"].unique()).issubset({"assembly", "stimulus_set"}):
        error = (
            f"The values of the 'lookup_type' column of the Catalog CSV file {path} MUST be"
            " either 'assembly' or 'stimulus_set'"
        )
        raise ValueError(error)


def validate_data_assembly(path: Path) -> None:
    """Validate a BrainIO Data Assembly.

    Ensures that the Data Assembly complies with the BrainIO specification.

    Args:
    ----
        path: path to the Data Assembly netCDF-4 file

    """
    assembly = xr.open_dataset(path)

    for required_attribute in ("identifier", "stimulus_set_identifier"):
        if required_attribute not in assembly.attrs:
            error = (
                f"'{required_attribute}' MUST be a global attribute of the Data Assembly"
                f" netCDF-4 file {path}"
            )
            raise ValueError(error)

        if not isinstance(assembly.attrs[required_attribute], str):
            error = (
                f"The '{required_attribute} global attribute of the Data Assembly netCDF-4"
                f" file {path} MUST be a string"
            )
            raise TypeError(error)


def validate_stimulus_set(*, path_csv: Path, path_zip: Path) -> None:
    """Validate a BrainIO Stimulus Set.

    Ensures that the Stimulus Set complies with the BrainIO specification.

    Args:
    ----
        path_csv: path to the Stimulus Set CSV file
        path_zip: path to the Stimulus Set ZIP file

    """
    if not (
        pd.read_csv(
            path_csv,
            nrows=1,
        ).columns.is_unique
    ):
        error = (
            f"The column headers of the Stimulus Set CSV file {path_csv} MUST be unique"
        )
        raise ValueError(error)

    file_csv = pd.read_csv(path_csv)
    for column in ("stimulus_id", "filename"):
        if column not in file_csv.columns:
            error = (
                f"'{column}' MUST be a column of the Stimulus Set CSV file {path_csv}"
            )
            raise ValueError(error)

        if not file_csv[column].is_unique:
            error = (
                f"The '{column}' column of the Stimulus Set CSV file {path_csv} MUST"
                " contain unique entries"
            )
            raise ValueError(error)

    for column in file_csv.columns:
        if not re.match(r"^[a-z0-9_]+$", column):
            error = (
                f"The column header '{column}' of the Stimulus Set CSV file {path_csv} MUST"
                " contain only lowercase alphabets, digits, and underscores"
            )
            raise ValueError(error)

    for stimulus_id in file_csv["stimulus_id"]:
        if not re.match(r"^[a-zA-z0-9]+$", stimulus_id):
            error = (
                f"The {stimulus_id} entry in the 'stimulus_id' column of the Stimulus Set"
                f" CSV file {path_csv} MUST be alphanumeric"
            )
            raise ValueError(error)

    with zipfile.ZipFile(path_zip, mode="r") as f:
        if not set(file_csv["filename"]).issubset(
            {zipinfo.filename for zipinfo in f.infolist()},
        ):
            error = (
                "All the filepaths in the 'filename' column of the Stimulus Set CSV file"
                f" {path_csv} MUST be present in the Stimulus Set ZIP archive {path_zip}"
            )
            raise ValueError(error)


def compute_sha1(path: Path) -> str:
    """Compute the SHA1 hash of a file.

    Args:
    ----
        path: path to the file

    Returns:
    -------
        SHA1 hash of the file as a hexdigest

    """
    buffer_size = 64 * 2**10
    sha1 = hashlib.sha1()  # noqa: S324
    with path.open("rb") as f:
        buffer = f.read(buffer_size)
        while len(buffer) > 0:
            sha1.update(buffer)
            buffer = f.read(buffer_size)
    return sha1.hexdigest()
