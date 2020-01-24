"""
The core interfaces of the yoga data layer.
"""

import abc
from typing import Any, Callable, Dict, Iterator, List


class Stream(metaclass=abc.ABCMeta):
    """
    The base class for objects returned by DataRef.stream().

    Conceptually, the core component of a Stream is an iterator factory that produce samples from
    the underlying dataset according to the arguments passed to DataRef.stream().
    """

    def __init__(
        self,
        iterator_factory: Callable[[], Iterator],
        epoch_len: int,
        stream_len: int,
        output_types: Any = None,
        output_shapes: Any = None,
    ):
        self._iterator_factory = iterator_factory
        self._epoch_len = epoch_len
        self._stream_len = stream_len
        self._output_types = output_types
        self._output_shapes = output_shapes

    def get_epoch_len(self) -> int:
        """
        Return the length of the epoch, or "how many records exist in the dataset".

        This will not match the actual number of records produced by this stream if options like
        sharding or a start offset.
        """
        return self._epoch_len

    def get_output_types(self) -> Any:
        """
        Return the output types stored with the dataset.

        This method is intended for integrating with the tf.data.Dataset API.
        """
        return self._output_types

    def get_output_shapes(self) -> Any:
        """
        Return the output shapes stored with the dataset.

        This method is intended for integrating with the tf.data.Dataset API.
        """
        return self._output_types

    def __len__(self) -> int:
        """
        Return the stream length, or "how many records will be produced by this stream".
        """
        return self._stream_len

    def __iter__(self) -> Iterator:
        return self._iterator_factory()


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
        shuffle_seed: int = -1,
        shard_size: int = 1,
        shard_rank: int = 0,
    ) -> Stream:
        """
        Produce a Stream object from a DataRef object.
        """
        pass

    @abc.abstractmethod
    def epoch_len(self) -> int:
        """
        All must be stored with metadata about how long they are.
        """
        pass


class Storage(metaclass=abc.ABCMeta):
    """
    Storage is a cache for datasets.

    Storage accepts datasets in various forms via submit(), and returns DataRef objects via
    fetch().

    Conceptually, Storage is sort of like a DataRef factory. It stores datasets
    in an unspecified format, and returns objects which implement the DataRef
    interface.

    Note that submission and cacheing are not multiprocessing-safe by default.
    There will be subclasses of Storage which provide such safety, either via
    lock files on a shared filesystem or via some outside worker. In either
    case, the @cacheable decorator should be safe to call simultaneously from
    many threads, processes, or machines.
    """

    @abc.abstractmethod
    def list_datasets(self) -> List[Dict[str, List[int]]]:
        """
        List the datsets and versions in this storage.
        """
        pass

    @abc.abstractmethod
    def submit(
        self, data: Any, dataset_id: str, dataset_version: int = 0, allow_overwrite: bool = False,
    ) -> None:
        """
        Iterate through a dataset and save it to storage.
        """
        pass

    @abc.abstractmethod
    def fetch(self, dataset_id: str, dataset_version: int = 0) -> DataRef:
        """
        Fetch a dataset from storage and provide a DataRef for streaming from it.
        """
        pass

    @abc.abstractmethod
    def cacheable(
        self, dataset_id: str, dataset_version: int = 0
    ) -> Callable[[Callable[[], Any]], Callable[[], DataRef]]:
        """
        A decorator which encompases calls to submit and fetch.

        The decorated function should return a dataset or a tuple of datasets. If the dataset is
        already cached, the function will not be executed.

        This method is capable of providing race-free access to the cache if multiple workers are
        attempting to submit-or-fetch the same dataset simultaneously.
        """
        pass
