__all__ = ["create_image_datapipe"]

from collections.abc import Callable, Sequence, Hashable

import numpy as np
import numpy.typing as npt
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper
from PIL import Image


def create_image_datapipe(
    datapipe: MapDataPipe,
    *,
    preprocess_fn: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
    indices: list[Hashable] = None,
) -> IterDataPipe:
    """Creates a PyTorch datapipe for loading images from disk and preprocessing them.

    Args:
        datapipe: source datapipe that maps a key to a PIL.Image.Image
        preprocess_fn: function used to preprocess each PIL.Image.Image
        batch_size: batch size

    Returns:
        torch datapipe that produces batches of data in the form (image_tensor, image_id)
    """

    def collate_fn(
        batch: Sequence[tuple[torch.Tensor, str]]
    ) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
        images = torch.stack([pair[0] for pair in batch])
        ids = np.array([pair[1] for pair in batch])
        return images, ids

    return (
        datapipe.zip(SequenceWrapper(indices))
        .map(fn=preprocess_fn)
        .to_iter_datapipe(indices=indices)
        .batch(batch_size=batch_size)
        .collate(collate_fn=collate_fn)
    )
