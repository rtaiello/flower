# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for model quantization."""


from typing import List, cast

import numpy as np

from flwr.common.typing import NDArrayFloat, NDArrayInt


CLIPPING_RANGE = 3
TARGET_RANGE = 2**16

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

def divide(xs, k):
        xs = np.array(xs, dtype=np.uint32)
        return (xs / k).tolist()


def reverse_quantize(
    weights: List[int],
    clipping_range: int = CLIPPING_RANGE,
    target_range: int = TARGET_RANGE,
) -> List[float]:
    max_range = clipping_range
    min_range = -clipping_range
    step_size = (max_range - min_range) / (target_range - 1)
    f = np.vectorize(lambda x: (min_range + step_size * x))

    weights = np.array(weights)
    reverse_quantized_list = f(weights.astype(np.float32))

    return reverse_quantized_list