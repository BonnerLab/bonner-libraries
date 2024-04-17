import xarray as xr


def concatenate_features(features: dict[str, xr.DataArray]) -> xr.DataArray:
    """Concatenates features from multiple nodes along the ``neuroid`` dimension.

    Args:
    ----
        features: dictionary of features from each node

    Returns:
    -------
        single DataArray containing features from all nodes

    """
    for node, feature in features.items():
        feature["node"] = ("neuroid", [node] * feature.sizes["neuroid"])

    return xr.concat(features.values(), dim="neuroid").rename("concatenated")


def flatten_features(features: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Flattens features from each node into a ``neuroid`` dimension.

    Args:
    ----
        features: dictionary of features from each node

    Returns:
    -------
        dictionary of flattened features from each node

    """
    for node in features:
        dims = list(set(features[node].dims) - {"presentation"})
        features[node] = features[node].stack({"neuroid": dims}).reset_index("neuroid")
    return features
