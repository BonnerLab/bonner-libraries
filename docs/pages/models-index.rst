Bonner Lab | Models
===================

User Guide
----------

Motivation
^^^^^^^^^^

``bonner-models`` aims to collate all the tools used by the Bonner Lab to dissect artificial neural network models implemented in PyTorch. Currently, we have standardized the extraction of activations from PyTorch models, making use of the latest PyTorch features.

Installation
^^^^^^^^^^^^

To use the development version of the package, run:

``pip install git+https://github.com/BonnerLab/bonner-models``

Since the package is under active development, we have not yet released a stable version.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

All extracted activations are stored within the directory specified by ``$BONNER_MODELS_HOME``.

Things to do
------------

- TODO use DataLoaderV2 to implement multiprocessing for loading data input
- TODO update test to use datapipe interface

API Reference
-------------

bonner.models
^^^^^^^^^^^^^

.. automodule:: bonner.models
   :ignore-module-all:
   :special-members: __init__
   :members:
   :private-members:
   :noindex:
   :undoc-members:
