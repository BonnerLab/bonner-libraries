User Guide
==========

Motivation
----------

The BrainIO format, originally developed by the `Brain-Score team <https://github.com/brain-score>`_, aims to "standardize the exchange of data between experimental and computational neuroscientists" by providing a minimal `specification <https://github.com/brain-score/brainio/blob/main/docs/SPECIFICATION.md>`_ for the data and `tools <https://github.com/brain-score/brainio>`_ for working with that data.

However, the reference implementation has some pain points, especially related to the handling of large netCDF-4 files that make it unsuitable for working with large-scale fMRI data. Additionally, though the specification has evolved, the tools have not yet kept pace and occasionally assume unspecified structure in the data.

To overcome these difficulties, we developed an in-house `minimal implementation <https://github.com/BonnerLab/bonner-brainio>`_ of the BrainIO specification primarily for use by members of the Bonner Lab, though we'd love for other labs to adopt its use and contribute too!

Installation
------------

To use the development version of the package, run:

``pip install git+https://github.com/BonnerLab/bonner-brainio``

Since the package is under active development, we have not yet released a stable version.

Environment variables
---------------------

All ``bonner-brainio`` data will be stored at the path specified by ``BONNER_BRAINIO_CACHE``.

Default directory structure
---------------------------

- Catalog CSV files are stored at ``$BONNER_BRAINIO_CACHE/<catalog-identifier>/catalog.csv``
- When loading assemblies and stimulus sets, the files are downloaded to ``$BONNER_BRAINIO_CACHE/<catalog-identifier>/``
- When packaging assemblies and stimulus sets using the convenience functions, the files are first placed in ``$BONNER_BRAINIO_CACHE/<catalog-identifier>/`` before being pushed to the specified remote location
