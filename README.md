# Bonner Lab | Libraries

## Libraries

- `bonner.brainio` - implementation of the BrainIO specification for neural datasets
- `bonner.files` - handling files
- `bonner.plotting` - plotting figures
- `bonner.datasets` - handling neural datasets
- `bonner.models` - working with artificial neural networks
- `bonner.computation` - CPU/GPU-agnostic computation
- `bonner.caching` - caching function outputs to disk

### `bonner.models`

`bonner-models` aims to collate all the tools used by the Bonner Lab to dissect artificial neural network models implemented in PyTorch. Currently, we have standardized the extraction of activations from PyTorch models, making use of the latest PyTorch features.


### BrainIO

The BrainIO format, originally developed by the `Brain-Score team <https://github.com/brain-score>`_, aims to "standardize the exchange of data between experimental and computational neuroscientists" by providing a minimal `specification <https://github.com/brain-score/brainio/blob/main/docs/SPECIFICATION.md>`_ for the data and `tools <https://github.com/brain-score/brainio>`_ for working with that data.

However, the reference implementation has some pain points, especially related to the handling of large netCDF-4 files that make it unsuitable for working with large-scale fMRI data. Additionally, though the specification has evolved, the tools have not yet kept pace and occasionally assume unspecified structure in the data.

- Catalog CSV files are stored at ``$BONNER_BRAINIO_HOME/<catalog-identifier>/catalog.csv``
- When loading assemblies and stimulus sets, the files are downloaded to ``$BONNER_BRAINIO_HOME/<catalog-identifier>/``
- When packaging assemblies and stimulus sets using the convenience functions, the files are first placed in ``$BONNER_BRAINIO_HOME/<catalog-identifier>/`` before being pushed to the specified remote location

## Environment variables

- `BONNER_BRAINIO_HOME`
- `BONNER_CACHING_HOME`
- `BONNER_DATASETS_HOME`
- `BONNER_MODELS_HOME`

## Credentials

To use the NSD, you will need to set the `AWS_SHARED_CREDENTIALS_FILE` environment variable, typically `~/.aws/credentials`
