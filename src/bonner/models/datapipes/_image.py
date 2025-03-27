from collections.abc import Callable, Hashable, Sequence

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, StackDataset

# def collate_fn(
#     batch: Sequence[tuple[torch.Tensor, str]],
# ) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
#     images = torch.stack([pair[0] for pair in batch])
#     ids = np.array([pair[1] for pair in batch])
#     return images, ids


def create_image_datapipe(
    datapipe: Dataset,
    *,
    preprocess_fn: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
    indices: list[Hashable] | None = None,
) -> DataLoader:
    # FIXME(Raj) deprecation of torchdata: https://github.com/pytorch/data/issues/1196
    """Create a PyTorch datapipe for loading images and preprocessing them.

    Args:
    ----
        datapipe: source datapipe that maps a key to a PIL.Image.Image
        preprocess_fn: function used to preprocess each PIL.Image.Image
        batch_size: batch size

    Returns:
    -------
        torch datapipe that produces batches of data in the form (image_tensor, image_id)

    """

    def collate_fn(
        batch: Sequence[tuple[torch.Tensor, str]],
    ) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
        images = torch.stack([preprocess_fn(pair[0]) for pair in batch])
        ids = np.array([pair[1] for pair in batch])
        return images, ids

    return DataLoader(
        StackDataset(datapipe, IterableDataset(indices)),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
