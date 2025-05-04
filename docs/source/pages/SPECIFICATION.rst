BrainIO Specification
=====================

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

BrainIO defines three types of entity:

* Data Assembly
* Stimulus Set
* Catalog

.. _specification.assembly:


Data Assembly
-------------

A Data Assembly MUST comprise

    * a string identifier
    * a `netCDF-4 <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#netcdf_4_spec>`_ file which

        * MUST have a root `group <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#groups_spec>`_ that contains exactly one `non-coordinate <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#dims_spec>`_ `variable <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#vars_spec>`_
        * MUST contain the string-type `global attributes <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#atts_spec>`_ 

            * ``identifier``, which MUST have a value identical to the Data Assembly identifier
            * ``stimulus_set_identifier``

.. _specification.stimulus_set:

Stimulus Set
------------

A Stimulus Set MUST comprise

* a `ZIP <https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>`_ archive
* a `CSV <https://datatracker.ietf.org/doc/html/rfc4180>`_ file

    * that MUST have a `header row <https://datatracker.ietf.org/doc/html/rfc4180>`__ of column names, which

        * MUST comprise lowercase letters, numerals, and underscores
        * MUST be unique within the row

    * that MUST contain columns named

        * ``filename``, whose entries

            * MUST contain the relative filepath of a file within the Stimulus Set ZIP archive
            * MUST be unique within the column

        * ``stimulus_id``, whose entries

            * MUST be alphanumeric 
            * MUST be unique within the column

.. _specification.catalog:

Catalog
-------

A Catalog

* MUST comprise a string identifier
* MAY comprise Data Assemblies and/or Stimulus Sets, where

    * each Data Assembly netCDF-4 file, Stimulus Set ZIP archive, and Stimulus Set CSV file MUST correspond to a row in the Catalog CSV file
    * each Data Assembly netCDF-4 file's ``stimulus_set_identifier`` `global attribute <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#atts_spec>`_ MUST correspond to the identifier of a Stimulus Set

* MUST comprise a `CSV file <https://datatracker.ietf.org/doc/html/rfc4180>`_

    * that MUST have a `header row <https://datatracker.ietf.org/doc/html/rfc2119>`_  of column names, which
        
        * MUST comprise lowercase letters, numerals, and underscores
        * MUST be unique within the row

    * where each row MUST correspond to a Data Assembly netCDF-4 file, Stimulus Set ZIP archive, or Stimulus Set CSV file in the Catalog
    * that MUST contain columns named

        * ``sha1``, whose entries

            * MUST be the `SHA1 hashes <https://datatracker.ietf.org/doc/html/rfc3174>`_ of the files corresponding to the rows
            * MUST be unique within the column

        * ``lookup_type``, whose entries

            * MUST be ``assembly`` if the rows corresponds to Data Assembly netCDF-4 files
            * MUST be ``stimulus_set`` if the rows correspond to Stimulus Set ZIP archives or Stimulus Set CSV files

        * ``identifier``, whose entries,

            * if the rows correspond to Data Assembly netCDF-4 files,

                * MUST be equal to the ``identifier`` `global attributes <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#atts_spec>`_ of the Data Assembly netCDF-4 files
                * MUST be unique among all the rows in the Catalog that correspond to Data Assembly netCDF-4 files

            * if the rows correspond to Stimulus Set ZIP archives or Stimulus Set CSV files,

                * MUST be equal to the identifiers of the Stimulus Sets

        * ``stimulus_set_identifier``, whose entries

            * MUST be equal to the ``stimulus_set_identifier`` `global attribute <https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#atts_spec>`_ of the Data Assembly if the row corresponds to a Data Assembly
            * MUST be empty if the row corresponds to a Stimulus Set ZIP Archive or a Stimulus Set CSV file

        * ``location_type``
        * ``location``
        * ``class``
