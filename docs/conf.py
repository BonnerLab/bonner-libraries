import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "Bonner Lab | Datasets"
copyright = "2022, Raj Magesh Gauthaman"
author = "Raj Magesh Gauthaman"
release = "0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx"]

autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "boto3": ("https://boto3.amazonaws.com/v1/documentation/api/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    # "netCDF4": ("http://unidata.github.io/netcdf4-python", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "nipy": ("http://nipy.org/nipy/api/", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "Bonner Lab | Datasets"
html_short_title = "Datasets"
