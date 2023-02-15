"""Utility functions for validating Catalogs, Data Assemblies, and Stimulus Sets."""

__all__: list[str] = []

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr


def validate_catalog(path: Path) -> None:
    """Validate a BrainIO Catalog.

    Ensures that the Catalog complies with the BrainIO specification*.

    * Warning: This does NOT check whether the Data Assemblies and Stimulus Sets stored in the Catalog are themselves valid.
    * Warning: This does NOT check whether the 'identifier' and 'stimulus_set_identifier' columns of a row corresponding to a Data Assembly netCDF-4 file match its global attributes of the same name.
    * Warning: This does NOT check whether the all the entries in the 'stimulus_set_identifier' column are present a

    Args:
        path: path to the Catalog CSV file
    """
    assert pd.read_csv(
        path,
        nrows=1,
        dtype=str,
    ).columns.is_unique, (
        f"The column headers of the Catalog CSV file {path} MUST be unique"
    )

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
        assert (
            required_column in catalog.columns
        ), f"'{required_column}' MUST be a column of the Catalog CSV file {path}"

    for column in catalog.columns:
        assert re.match(r"^[a-z0-9_]+$", column), (
            f"The column header '{column}' of the Catalog CSV file {path} MUST contain"
            " only lowercase alphabets, digits, and underscores"
        )

    if catalog.empty:
        return

    for column in {"identifier", "stimulus_set_identifier"}:
        assert catalog.dtypes[column] == "O", (
            f"The column '{column}' of the Catalog CSV file {path} MUST have string"
            " entries"
        )

    for sha1 in catalog["sha1"]:
        assert (
            re.match(r"^[a-fA-F0-9]+$", sha1) and len(sha1) == 40
        ), f"The SHA1 hash {sha1} in the Catalog CSV file {path} is invalid"

    assert (
        catalog.loc[catalog["lookup_type"] == "assembly"]
        .groupby("identifier")
        .count()["sha1"]
        == 1
    ).all(), (
        "Each Data Assembly MUST have exactly 1 corresponding row in the Catalog CSV"
        f" file {path}"
    )
    assert (
        catalog.loc[catalog["lookup_type"] == "stimulus_set"]
        .groupby("identifier")
        .count()["sha1"]
        == 2
    ).all(), (
        "Each Stimulus Set MUST have exactly 2 corresponding rows in the Catalog CSV"
        f" file {path}"
    )

    assert catalog[
        "sha1"
    ].is_unique, (
        f"The 'sha1' column of the Catalog CSV file {path} MUST contain unique entries"
    )

    assert set(catalog["lookup_type"].unique()).issubset(
        {"assembly", "stimulus_set"}
    ), (
        f"The values of the 'lookup_type' column of the Catalog CSV file {path} MUST be"
        " either 'assembly' or 'stimulus_set'"
    )


def validate_data_assembly(path: Path) -> None:
    """Validate a BrainIO Data Assembly.

    Ensures that the Data Assembly complies with the BrainIO specification.

    Args:
        path: path to the Data Assembly netCDF-4 file
    """

    assembly = xr.open_dataset(path)

    for required_attribute in {"identifier", "stimulus_set_identifier"}:
        assert required_attribute in assembly.attrs, (
            f"'{required_attribute}' MUST be a global attribute of the Data Assembly"
            f" netCDF-4 file {path}"
        )

        assert isinstance(assembly.attrs[required_attribute], str), (
            f"The '{required_attribute} global attribute of the Data Assembly netCDF-4"
            f" file {path} MUST be a string"
        )


def validate_stimulus_set(*, path_csv: Path, path_zip: Path) -> None:
    """Validate a BrainIO Stimulus Set.

    Ensures that the Stimulus Set complies with the BrainIO specification.

    Args:
        path_csv: path to the Stimulus Set CSV file
        path_zip: path to the Stimulus Set ZIP file
    """
    assert pd.read_csv(
        path_csv, nrows=1
    ).columns.is_unique, (
        f"The column headers of the Stimulus Set CSV file {path_csv} MUST be unique"
    )

    file_csv = pd.read_csv(path_csv)
    for column in {"stimulus_id", "filename"}:
        assert (
            column in file_csv.columns
        ), f"'{column}' MUST be a column of the Stimulus Set CSV file {path_csv}"

        assert file_csv[column].is_unique, (
            f"The '{column}' column of the Stimulus Set CSV file {path_csv} MUST"
            " contain unique entries"
        )

    for column in file_csv.columns:
        assert re.match(r"^[a-z0-9_]+$", column), (
            f"The column header '{column}' of the Stimulus Set CSV file {path_csv} MUST"
            " contain only lowercase alphabets, digits, and underscores"
        )

    for stimulus_id in file_csv["stimulus_id"]:
        assert re.match(r"^[a-zA-z0-9]+$", stimulus_id), (
            f"The {stimulus_id} entry in the 'stimulus_id' column of the Stimulus Set"
            f" CSV file {path_csv} MUST be alphanumeric"
        )

    with zipfile.ZipFile(path_zip, mode="r") as f:
        assert set(file_csv["filename"]).issubset(
            {zipinfo.filename for zipinfo in f.infolist()}
        ), (
            "All the filepaths in the 'filename' column of the Stimulus Set CSV file"
            f" {path_csv} MUST be present in the Stimulus Set ZIP archive {path_zip}"
        )


def compute_sha1(path: Path) -> str:
    """Compute the SHA1 hash of a file.

    Args:
        path: path to the file

    Returns:
        SHA1 hash of the file as a hexdigest
    """
    buffer_size = 64 * 2**10
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        buffer = f.read(buffer_size)
        while len(buffer) > 0:
            sha1.update(buffer)
            buffer = f.read(buffer_size)
    return sha1.hexdigest()
