import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Self

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
from torchvision import transforms

DEFAULT_CURVATURES = np.logspace(-2, -0.1, 5)


def create_curvature_filters(
    n_orientations: int = 16,
    curvatures: np.ndarray = DEFAULT_CURVATURES,
    gaussian_sizes: Sequence[int] = (5,),
    filter_size: int = 9,
    frequencies: Sequence[float] = [1.2],
    gamma: float = 1,
    sigma_x: float = 1,
    sigma_y: float = 1,
) -> torch.Tensor:
    orientations = np.arange(0, 2 * np.pi, 2 * np.pi / n_orientations)

    weights = [
        create_banana_filter(
            size=gaussian_size,
            frequency=frequency,
            theta=orientation,
            curvature=curvature,
            gamma=gamma,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            filter_size=filter_size,
        )
        for curvature, gaussian_size, orientation, frequency in itertools.product(
            curvatures,
            gaussian_sizes,
            orientations,
            frequencies,
        )
    ]
    return torch.stack(weights).unsqueeze(1)


def create_banana_filter(
    *,
    size: float,
    frequency: float,
    theta: float,
    curvature: float,
    gamma: float,
    sigma_x: float,
    sigma_y: float,
    filter_size: int,
) -> torch.Tensor:
    xv, yv = np.meshgrid(
        np.arange(
            np.fix(-filter_size / 2).item(),
            np.fix(filter_size / 2).item() + filter_size % 2,
        ),
        np.arange(
            np.fix(filter_size / 2).item(),
            np.fix(-filter_size / 2).item() - filter_size % 2,
            -1,
        ),
    )
    xv = xv.T
    yv = yv.T

    # define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # define the bias term
    bias = np.exp(-sigma_x / 2)
    k = xc + curvature * (xs**2)

    # define the rotated Guassian rotated and curved function
    k2 = (k / sigma_x) ** 2 + (xs / (sigma_y * size)) ** 2
    g = np.exp(-k2 * frequency**2 / 2)

    # define the rotated and curved complex wave function
    f = np.exp(frequency * k * 1j)

    # multiply the complex wave function with the Gaussian function
    # with a constant and bias
    filter_ = gamma * g * (f - bias)
    filter_ = np.real(filter_)
    filter_ -= filter_.mean()

    return torch.from_numpy(filter_).float()


def create_random_filters(
    out_channels: int,
    in_channels: int,
    kernel_size: int,
    *,
    smooth: bool = False,
    seed: int = 27,
) -> torch.Tensor:
    torch.manual_seed(seed)
    weights = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    weights -= weights.mean(dim=[2, 3], keepdim=True)  # mean centering
    if smooth:
        num_smoothed = round(out_channels * 0.5)
        idx_smoothed = random.sample(list(np.arange(0, out_channels)), num_smoothed)
        for i in idx_smoothed:
            channel_smoothed = torch.Tensor(
                gaussian_filter(weights[i, :, :, :], sigma=1),
            )
            weights[i, :, :, :] = channel_smoothed

    return weights


class AtlasNet(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=45,
            padding=math.floor(45 / 2),
        )
        self.maxpool1 = nn.MaxPool2d(6)
        self.conv2 = nn.Conv2d(
            in_channels=24,
            out_channels=20000,
            kernel_size=9,
            padding=math.floor(9 / 2),
        )
        self.maxpool2 = nn.MaxPool2d(8)
        self.flatten = nn.Flatten()

        self.conv1.data = create_curvature_filters(
            n_orientations=3,
            curvatures=np.logspace(-2, -0.1, 8),
            gaussian_sizes=(5,),
            filter_size=45,
            frequencies=[1.2],
        )
        self.conv2.data = create_random_filters(
            out_channels=20000,
            in_channels=24,
            kernel_size=9,
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        return self.flatten(x)


def preprocess(image: Image.Image) -> torch.Tensor:
    size = 96
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ],
    )
    return transform(image.convert("RGB"))


def load() -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    return AtlasNet(), preprocess
