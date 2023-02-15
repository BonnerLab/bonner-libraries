import xarray as xr


def concatenate_features(features: dict[str, xr.DataArray]) -> xr.DataArray:
    """Concatenates features from multiple nodes along the ``neuroid`` dimension.

    Args:
        features: dictionary of features from each node

    Returns:
        single DataArray containing features from all nodes
    """
    for node, feature in features.items():
        feature["node"] = ("neuroid", [node] * feature.sizes["neuroid"])

    return xr.concat(features.values(), dim="neuroid").rename("concatenated")
