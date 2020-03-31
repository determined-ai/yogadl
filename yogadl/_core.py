# Copyright 2020 Determined AI. All Rights Reserved.
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
"""
The core interfaces of the yoga data layer.
"""

import abc
from typing import Any, Callable, Optional, Union

import tensorflow as tf


# TODO: Make sure users are not required to have TF, PyTorch,
# and TP dataflows all installed to use this.
Submittable = Union[
    tf.data.Dataset,
]


class Stream:
    """
    Stream contains a generator of data and other required information
    to feed into framework specific data APIs.
    """

    def __init__(
        self,
        iterator_fn: Callable,
        length: int,
        output_types: Any = None,
        output_shapes: Any = None,
    ):
        self.iterator_fn = iterator_fn
        self.length = length
        self.output_types = output_types
        self.output_shapes = output_shapes

    def __iter__(self) -> Any:
        return self.iterator_fn()

    def __len__(self) -> int:
        return self.length


class DataRef(metaclass=abc.ABCMeta):
    """
    The base interface for a reference to a dataset in the yogadl framework.

    The DataRef may refer to a dataset in a remote storage location; it need not refer to locally-
    available data. The only mechanism for accessing the records inside the dataset is to create a
    Stream and to iterate through them.

    By specifying all of the random-access options up front, the backend which provides the DataRef
    can provide performance-optimized streaming, since it is guaranteed with yogadl that lower
    layers will operate without random access.
    """

    @abc.abstractmethod
    def stream(
        self,
        start_offset: int = 0,
        shuffle: bool = False,
        skip_shuffle_at_epoch_end: bool = False,
        shuffle_seed: Optional[int] = None,
        shard_rank: int = 0,
        num_shards: int = 1,
        drop_shard_remainder: bool = False,
    ) -> Stream:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class Storage(metaclass=abc.ABCMeta):
    """
    Storage is a cache for datasets.

    Storage accepts datasets in various forms via submit(), and returns DataRef objects via
    fetch().

    Conceptually, Storage is sort of like a DataRef factory. It stores datasets
    in an unspecified format, and returns objects which implement the DataRef
    interface.

    Note that submit() and fetch() are not multiprocessing-safe by default.
    The @cacheable decorator should be safe to call simultaneously from
    many threads, processes, or machines.
    """

    @abc.abstractmethod
    def submit(self, data: Submittable, dataset_id: str, dataset_version: str) -> None:
        """
        Stores dataset to a cache.
        """
        pass

    @abc.abstractmethod
    def fetch(self, dataset_id: str, dataset_version: str) -> DataRef:
        """
        Fetch a dataset from storage and provide a DataRef for streaming it.
        """
        pass

    @abc.abstractmethod
    def cacheable(self, dataset_id: str, dataset_version: str) -> Callable:
        """
        A decorator that calls submit and fetch and is responsible for coordinating
        amongst instances of Storage in different processes.
        """
        pass
