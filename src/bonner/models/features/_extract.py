from pathlib import Path

from tqdm.auto import tqdm
import netCDF4
import xarray as xr
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchdata.datapipes.iter import IterDataPipe

from bonner.models.hooks import Hook
from bonner.models.utilities import BONNER_MODELS_HOME


def extract_features(
    *,
    model: torch.nn.modules.module.Module,
    model_identifier: str,
    nodes: list[str],
    hooks: dict[str, Hook],
    datapipe: IterDataPipe,
    datapipe_identifier: str,
    cache_path: Path = BONNER_MODELS_HOME,
    use_cached: bool = True,
    device: torch.device | None = None,
) -> dict[str, xr.DataArray]:
    """Extract features from the internal nodes of a PyTorch model.

    WARNING: this function assumes that

    * all 4-D features are from convolutional layers and have the shape ``(presentation, channel, spatial_x, spatial_y)``
    * all 2-D features are from linear layers and have the shape ``(presentation, channel)``
    * all 3-D features are from patch-based Vision Transformers and have the shape ``(presentation, patch, channel)``

    Args:
        model: a PyTorch model
        model_identifier: identifier for the model
        nodes: list of layer names to extract features from, in standard PyTorch format (e.g. 'classifier.0')
        hooks: dictionary mapping layer names to hooks to be applied to the features extracted from the layer (e.g. {"conv2": GlobalMaxpool()})
        datapipe: torch datapipe that provides batches of data of the form ``(data, stimulus_ids)``. ``data`` is a torch Tensor with shape (batch_size, *) and ``stimulus_ids`` is a Numpy array of string identifiers corresponding to each stimulus in ``data``.
        datapipe_identifier: identifier for the dataset
        use_cached: whether to use previously computed features, defaults to True
        device: torch device on which the feature extraction will occur, defaults to None

    Returns:
        dictionary where keys are node identifiers and values are xarray DataArrays containing the model's features. Each ``xarray.DataArray`` has a ``presentation`` dimension corresponding to the stimuli with a ``stimulus_id`` coordinate corresponding to the ``stimulus_ids`` from ``datapipe``, and other dimensions that depend on the layer type and the hook.
    """

    device = _get_device(device)
    cache_dir = _create_cache_directory(
        cache_path=cache_path,
        model_identifier=model_identifier,
        datapipe_identifier=datapipe_identifier,
    )
    filepaths = _get_filepaths(nodes=nodes, hooks=hooks, cache_dir=cache_dir)
    nodes_to_compute = _get_nodes_to_compute(
        nodes=nodes, filepaths=filepaths, use_cached=use_cached
    )

    if nodes_to_compute:
        _extract_features(
            model=model,
            nodes=nodes_to_compute,
            hooks=hooks,
            filepaths=filepaths,
            device=device,
            datapipe=datapipe,
        )

    assemblies = _open_with_xarray(
        model_identifier=model_identifier,
        filepaths=filepaths,
        hooks=hooks,
        datapipe_identifier=datapipe_identifier,
    )
    return assemblies


def _open_with_xarray(
    *,
    model_identifier: str,
    filepaths: dict[str, Path],
    hooks: dict[str, Hook],
    datapipe_identifier: str,
) -> dict[str, xr.DataArray]:
    assemblies = {}
    for node, filepath in filepaths.items():
        hook = hooks[node].identifier if node in hooks.keys() else None
        assembly = xr.open_dataset(filepath)
        assemblies[node] = (
            assembly[node]
            .assign_coords(
                {"stimulus_id": ("presentation", assembly["stimulus_id"].values)}
            )
            .rename(f"{model_identifier}.node={node}.hook={hook}.{datapipe_identifier}")
        )
    return assemblies


def _extract_features(
    *,
    model: torch.nn.modules.module.Module,
    nodes: list[str],
    hooks: dict[str, Hook],
    filepaths: dict[str, Path],
    device: torch.device,
    datapipe: IterDataPipe,
) -> None:
    netcdf4_files = {
        node: netCDF4.Dataset(filepaths[node], "w", format="NETCDF4") for node in nodes
    }

    with torch.no_grad():
        model = model.eval()
        model = model.to(device=device)

        extractor = create_feature_extractor(model, return_nodes=nodes)
        extractor = extractor.to(device=device)

        start = 0
        for batch, (input_data, stimulus_ids) in enumerate(
            tqdm(datapipe, desc="batch", leave=False)
        ):
            input_data = input_data.to(device)
            features = extractor(input_data)
            for node in features.keys():
                if node in hooks.keys():
                    features[node] = hooks[node](features[node])

            for node, netcdf4_file in netcdf4_files.items():
                features_node = features[node].detach().cpu().numpy()

                if batch == 0:
                    _create_netcdf4_file(
                        file=netcdf4_file,
                        node=node,
                        features=features_node,
                    )

                end = start + len(input_data)
                netcdf4_file.variables[node][start:end, ...] = features_node
                netcdf4_file.variables["presentation"][start:end] = (
                    np.arange(len(input_data)) + start
                )
                netcdf4_file.variables["stimulus_id"][start:end] = stimulus_ids

            start += len(input_data)

    for netcdf4_file in netcdf4_files.values():
        netcdf4_file.sync()
        netcdf4_file.close()


def _create_netcdf4_file(
    *,
    file: netCDF4.Dataset,
    node: str,
    features: torch.Tensor,
) -> None:
    match features.ndim:
        case 4:  # ASSUMPTION: convolutional layer
            dimensions = ["presentation", "channel", "spatial_x", "spatial_y"]
        case 2:  # ASSUMPTION: linear layer
            dimensions = ["presentation", "channel"]
        case 3:  # ASSUMPTION: patch-based ViT
            dimensions = ["presentation", "patch", "channel"]
        case _:
            raise ValueError("features do not have the appropriate shape")

    for dimension, length in zip(dimensions, (None, *features.shape[1:])):
        file.createDimension(dimension, length)
        if dimension == "presentation":
            file.createVariable(dimension, np.int64, (dimension,))
            file.createVariable("stimulus_id", str, (dimension,))
        else:
            variable = file.createVariable(dimension, np.int64, (dimension,))
            variable[:] = np.arange(length)

    dtype = np.dtype(getattr(np, str(features.dtype).replace("torch.", "")))
    file.createVariable(node, dtype, dimensions)


def _get_device(device: str | torch.device = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            device_ = torch.device("cuda")
        else:
            device_ = torch.device("cpu")
    else:
        device_ = torch.device(device)
    return device_


def _create_cache_directory(
    *, cache_path: Path, model_identifier: str, datapipe_identifier: str
) -> Path:
    cache_dir = (
        cache_path
        / "features"
        / f"{model_identifier}"
        / f"datapipe={datapipe_identifier}"
    )
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _get_filepaths(
    *,
    nodes: list[str],
    hooks: dict[str, Hook],
    cache_dir: Path,
) -> dict[str, Path]:
    filepaths: dict[str, Path] = {}
    for node in nodes:
        if node in hooks.keys():
            hook_identifier = hooks[node].identifier
        else:
            hook_identifier = "None"
        filepaths[node] = cache_dir / f"node={node}" / f"hook={hook_identifier}.nc"
        filepaths[node].parent.mkdir(exist_ok=True, parents=True)
    return filepaths


def _get_nodes_to_compute(
    *, nodes: list[str], filepaths: dict[str, Path], use_cached: bool
) -> list[str]:
    nodes_to_compute = nodes.copy()
    for node in nodes:
        if filepaths[node].exists():
            if use_cached:
                nodes_to_compute.remove(node)  # don't re-compute
            else:
                filepaths[node].unlink()  # delete pre-cached features
    return nodes_to_compute
