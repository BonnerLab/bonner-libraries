import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr
from loguru import logger

from bonner.brainio._network import fetch, send
from bonner.brainio._utilities import (
    BONNER_BRAINIO_HOME,
    compute_sha1,
    validate_catalog,
    validate_data_assembly,
    validate_stimulus_set,
)


class Catalog:
    def __init__(
        self,
        identifier: str = "bonner-brainio",
        *,
        csv_file: Path | None = None,
        cache_directory: Path | None = None,
        validate: bool = True,
    ) -> None:
        """Initialize a Catalog.

        Args:
            identifier: identifier of the Catalog
            csv_file: path to the (potentially existing) Catalog CSV file
            cache_directory: directory to use as a local file cache
        """
        self.identifier = identifier
        """Identifier of the Catalog."""

        self.csv_file: Path
        """Path to the Catalog CSV file, defaults to ``$BONNER_BRAINIO_HOME/<identifier>/catalog.csv``."""

        self.cache_directory: Path
        """Local cache directory for files fetched from the Catalog, defaults to ``$BONNER_BRAINIO_HOME/<identifier>``."""

        if csv_file:
            self.csv_file = csv_file
        else:
            self.csv_file = BONNER_BRAINIO_HOME / self.identifier / "catalog.csv"

        if not self.csv_file.exists():
            self._create(path=self.csv_file)

        if cache_directory:
            self.cache_directory = cache_directory
        else:
            self.cache_directory = BONNER_BRAINIO_HOME / self.identifier

        if not self.cache_directory.exists():
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        if validate:
            validate_catalog(path=self.csv_file)

    def load_stimulus_set(
        self,
        *,
        identifier: str,
        use_cached: bool = True,
        check_integrity: bool = True,
        validate: bool = True,
    ) -> dict[str, Path]:
        """Load a Stimulus Set from the Catalog.

        Args:
            identifier: identifier of the Stimulus Set
            use_cached: whether to use the local cache, defaults to True
            check_integrity: whether to check the SHA1 hashes of the files, defaults to True
            validate: whether to ensure that the Stimulus Set conforms to the BrainIO specification, defaults to True

        Returns:
            paths to the Stimulus Set CSV file and ZIP archive, with keys "csv" and "zip" respectively
        """
        metadata = self.lookup(identifier=identifier, lookup_type="stimulus_set")
        assert not metadata.empty, f"Stimulus Set {identifier} not found in Catalog"

        paths = {}
        for row in metadata.itertuples():
            path = fetch(
                path_cache=self.cache_directory,
                location_type=row.location_type,
                location=row.location,
                use_cached=use_cached,
            )

            if check_integrity:
                assert row.sha1 == compute_sha1(
                    path
                ), f"SHA1 hash from the Catalog does not match that of {path}"

            if zipfile.is_zipfile(path):
                paths["zip"] = path
            else:
                paths["csv"] = path

        if validate:
            validate_stimulus_set(path_csv=paths["csv"], path_zip=paths["zip"])

        return paths

    def load_data_assembly(
        self,
        *,
        identifier: str,
        use_cached: bool = True,
        check_integrity: bool = True,
        validate: bool = True,
    ) -> Path:
        """Load a Data Assembly from the Catalog.

        Args:
            identifier: identifier of the Data Assembly
            use_cached: whether to use the local cache, defaults to True
            check_integrity: whether to check the SHA1 hashes of the files, defaults to True
            validate: whether to ensure that the Data Assembly conforms to the BrainIO specification, defaults to True

        Returns:
            path to the Data Assembly netCDF-4 file
        """
        metadata = self.lookup(identifier=identifier, lookup_type="assembly")
        assert not metadata.empty, f"Data Assembly {identifier} not found in Catalog"

        path = fetch(
            path_cache=self.cache_directory,
            location_type=metadata["location_type"].item(),
            location=metadata["location"].item(),
            use_cached=use_cached,
        )

        if check_integrity:
            assert metadata["sha1"].item() == compute_sha1(
                path
            ), f"SHA1 hash from the Catalog does not match that of {path}"

        if validate:
            validate_data_assembly(path=path)

        return path

    def package_stimulus_set(
        self,
        *,
        identifier: str,
        path_csv: Path,
        path_zip: Path,
        location_type: str,
        location_csv: str,
        location_zip: str,
        class_csv: str,
        class_zip: str,
        force: bool = False,
    ) -> None:
        """Add a Stimulus Set to the Catalog.

        Args:
            identifier: identifier of the Stimulus Set
            path_csv: path to the Stimulus Set CSV file
            path_zip: path to the Stimulus Set ZIP file
            location_type: location_type of the Stimulus Set
            location_csv: remote URL of the Stimulus Set CSV file
            location_zip: remote URL of the Stimulus Set ZIP archive
            class_csv: class of the Stimulus Set CSV file
            class_zip: class of the Stimulus Set ZIP archive
            force: whether to repackage the Stimulus Set if it already exists, defaults to False
        """
        metadata = self.lookup(identifier=identifier, lookup_type="stimulus_set")
        if not (metadata.empty or force):
            logger.debug(
                f"Stimulus Set {identifier} exists in Catalog {self.identifier}, not"
                " re-packaging"
            )
            return

        if force:
            self._delete(identifier=identifier, lookup_type="stimulus_set")

        logger.debug(
            f"Packaging Stimulus Set {identifier} to Catalog {self.identifier}"
        )

        validate_stimulus_set(path_csv=path_csv, path_zip=path_zip)

        for path, location, class_ in {
            (path_csv, location_csv, class_csv),
            (path_zip, location_zip, class_zip),
        }:
            send(path=path, location_type=location_type, location=location)
            self._append(
                {
                    "identifier": identifier,
                    "lookup_type": "stimulus_set",
                    "class": class_,
                    "location_type": location_type,
                    "location": location,
                    "sha1": compute_sha1(path),
                    "stimulus_set_identifier": "",
                }
            )
        validate_catalog(self.csv_file)

    def package_data_assembly(
        self,
        *,
        path: Path,
        location_type: str,
        location: str,
        class_: str,
        force: bool = False,
    ) -> None:
        """Add a Data Assembly to the Catalog.

        Args:
            path: path to the Data Assembly netCDF-4 file
            location_type: location_type of the Data Assembly
            location: remote URL of the Data Assembly
            class_: class of the Data Assembly
            force: whether to repackage the Data Assembly if it already exists, defaults to False
        """
        validate_data_assembly(path=path)

        assembly = xr.open_dataset(path)
        identifier = assembly.attrs["identifier"]

        metadata = self.lookup(identifier=identifier, lookup_type="assembly")
        if not (metadata.empty or force):
            logger.debug(
                f"Data Assembly {identifier} exists in Catalog {self.identifier}, not"
                " re-packaging"
            )
            return

        if force:
            self._delete(identifier=identifier, lookup_type="assembly")

        logger.debug(
            f"Packaging Data Assembly {identifier} to Catalog {self.identifier}"
        )

        send(path=path, location_type=location_type, location=location)

        self._append(
            {
                "identifier": identifier,
                "lookup_type": "assembly",
                "class": class_,
                "location_type": location_type,
                "location": location,
                "sha1": compute_sha1(path),
                "stimulus_set_identifier": assembly.attrs["stimulus_set_identifier"],
            }
        )
        validate_catalog(self.csv_file)

    def _create(self, path: Path) -> None:
        """Create a new Catalog CSV file.

        Args:
            path: path where the Catalog CSV file should be created
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        catalog = pd.DataFrame(
            data=None,
            columns=(
                "identifier",
                "lookup_type",
                "sha1",
                "location_type",
                "location",
                "stimulus_set_identifier",
                "class",
            ),
        )
        catalog.to_csv(path, index=False)

    def lookup(
        self,
        *,
        identifier: str,
        lookup_type: str,
    ) -> pd.DataFrame:
        """Look up the metadata for a Data Assembly or Stimulus Set in the Catalog.

        Args:
            identifier: identifier of the Data Assembly or Stimulus Set
            lookup_type: 'assembly' or 'stimulus_set', when looking up Data Assemblies or Stimulus Sets respectively

        Returns:
            metadata corresponding to the Data Assembly or Stimulus Set
        """
        catalog = pd.read_csv(self.csv_file)
        filter_ = (catalog["identifier"] == identifier) & (
            catalog["lookup_type"] == lookup_type
        )
        return catalog.loc[filter_, :]

    def _append(self, entry: dict[str, str]) -> None:
        """Append an entry to the Catalog.

        Args:
            entry: a row to be appended to the Catalog CSV file, where keys correspond to column header names
        """
        catalog = pd.read_csv(self.csv_file)
        catalog = pd.concat([catalog, pd.DataFrame(entry, index=[len(catalog)])])
        catalog.to_csv(self.csv_file, index=False)

    def _delete(self, *, identifier: str, lookup_type: str) -> None:
        catalog = pd.read_csv(self.csv_file)
        filter_ = (catalog["identifier"] == identifier) & (
            catalog["lookup_type"] == lookup_type
        )
        catalog.loc[~filter_, :].to_csv(self.csv_file, index=False)
