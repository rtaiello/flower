from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIPPING_RANGE = 3
TARGET_RANGE = 2**16


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def quantize(
    weights: List[float],
    clipping_range: int = CLIPPING_RANGE,
    target_range: int = TARGET_RANGE,
) -> List[int]:
    f = np.vectorize(
        lambda x: min(
            target_range - 1,
            (sorted((-clipping_range, x, clipping_range))[1] + clipping_range)
            * target_range
            / (2 * clipping_range),
        )
    )
    quantized_list = f(weights).astype(int)

    return quantized_list.tolist()


def multiply(xs, k):
    """
    Multiplies a list of integers by a constant

    Args:
        xs: List of integers
        k: Constant to multiply by

    Returns:
        List of multiplied integers
    """
    xs = np.array(xs, dtype=np.uint32)
    return (xs * k).tolist()



