__all__ = ["create_image_datapipe"]

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from PIL import Image


def create_image_datapipe(
    image_paths: Sequence[Path],
    image_ids: Sequence[str],
    preprocess_fn: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
) -> IterDataPipe:
    """Creates a PyTorch datapipe for loading images from disk and preprocessing them.

    Args:
        image_paths: paths to the image files on disk
        image_ids: unique string identifiers for each image file
        preprocess_fn: function used to preprocess each Image
        batch_size: batch_size used for batching

    Todo:
        deprecate batch_size parameter once PyTorch DataloaderV2 is released)

    Raises:
        FileNotFoundError: if the specified image file is not found
        ValueError: string identifiers should be unique across images

    Returns:
        torch datapipe that produces batches of data in the form (image_tensor, image_id)
    """
    image_paths = [Path(path) for path in image_paths]
    for path in image_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"stimulus_path {path} does not exist")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("'stimulus_ids' must contain unique elements")

    def map_fn(path: Path) -> torch.Tensor:
        with Image.open(path) as f:
            result = preprocess_fn(f)
        return result

    def collate_fn(
        batch: Sequence[tuple[torch.Tensor, str]]
    ) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
        images = torch.stack([pair[0] for pair in batch])
        ids = np.array([pair[1] for pair in batch])
        return images, ids

    return (
        IterableWrapper(image_paths)
        .map(fn=map_fn)
        .zip(IterableWrapper(image_ids))
        .batch(batch_size=batch_size)
        .collate(collate_fn=collate_fn)
    )
