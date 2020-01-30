import logging
import pathlib
import pickle
import platform
from typing import Any, cast, Generator, List

import lmdb


def serialize_generator_to_lmdb(
    dataset_generator: Generator,
    data_shapes: Any,
    data_types: Any,
    checkpoint_path: pathlib.Path,
    write_frequency: int = 5000,
) -> int:
    """
    Serialize a generator to a single LMDB file. Adapted from [1].

    [1] https://tensorpack.readthedocs.io/_modules/tensorpack/dataflow/serialize.html
    """
    assert checkpoint_path.parent.is_dir(), "Checkpoint directory does not exist."
    assert not checkpoint_path.exists(), "Checkpoint path already exists."
    # It's OK to use super large map_size on Linux, but not on other platforms
    # See: https://github.com/NVIDIA/DIGITS/issues/206
    map_size = 1099511627776 * 2 if platform.system() == "Linux" else 128 * 10 ** 6
    db = lmdb.open(
        str(checkpoint_path),
        subdir=False,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )  # need sync() at the end

    # put data into lmdb, and doubling the size if full.
    # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
    def put_or_grow(txn: lmdb.Transaction, key: Any, value: Any) -> lmdb.Transaction:
        try:
            txn.put(key, value)
            return txn
        except lmdb.MapFullError:
            pass
        txn.abort()
        curr_size = db.info()["map_size"]
        new_size = curr_size * 2
        logging.info(f"Doubling LMDB map_size to {new_size / 10 ** 9} GB")
        db.set_mapsize(new_size)
        txn = db.begin(write=True)
        txn = put_or_grow(txn, key, value)
        return txn

    # LMDB transaction is not exception-safe!
    # although it has a context manager interface
    txn = db.begin(write=True)
    dataset_entries = 0
    while True:
        try:
            data = next(dataset_generator)
        except StopIteration:
            break
        txn = put_or_grow(
            txn=txn,
            key="{:08}".format(dataset_entries).encode("ascii"),
            value=pickle.dumps(data, protocol=-1),
        )
        if dataset_entries % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
        dataset_entries += 1
    txn.commit()

    keys = ["{:08}".format(k).encode("ascii") for k in range(dataset_entries)]
    with db.begin(write=True) as txn:
        put_or_grow(txn=txn, key=b"__keys__", value=pickle.dumps(keys, protocol=-1))
        put_or_grow(txn=txn, key=b"__shapes__", value=pickle.dumps(data_shapes, protocol=-1))
        put_or_grow(txn=txn, key=b"__types__", value=pickle.dumps(data_types, protocol=-1))

    logging.debug("Flushing database ...")
    db.sync()
    db.close()

    return dataset_entries


class LmdbAccess:
    """
    Provides random access to an LMDB store file. Adopted from [1].

    [1] https://github.com/tensorpack/tensorpack/blob/master/tensorpack/dataflow/format.py
    """

    def __init__(self, lmdb_path: pathlib.Path) -> None:
        assert lmdb_path.exists()
        self._lmdb_path = lmdb_path
        self._db_connection_open = False

        self._open_lmdb()
        self._size = cast(int, self._txn.stat()["entries"])
        self._read_keys_from_db()
        self._read_shapes_from_db()
        self._read_types_from_db()
        logging.info("Found {} entries in {}".format(self._size, self._lmdb_path))

        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdb()

    def __exit__(self, *_: Any) -> None:
        self._close_lmdb()

    def _open_lmdb(self) -> None:
        self._lmdb = lmdb.open(
            str(self._lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            map_size=1099511627776 * 2,
            max_readers=100,
        )
        self._txn = self._lmdb.begin()
        self._db_connection_open = True

    def _read_keys_from_db(self) -> None:
        self._keys = self._txn.get(b"__keys__")
        assert self._keys is not None
        self._keys = cast(List[bytes], pickle.loads(self._keys))
        self._size -= 1  # delete this item

    def _read_shapes_from_db(self) -> None:
        self._shapes = self._txn.get(b"__shapes__")
        assert self._shapes is not None
        self._shapes = pickle.loads(self._shapes)

    def _read_types_from_db(self) -> None:
        self._types = self._txn.get(b"__types__")
        assert self._types is not None
        self._types = pickle.loads(self._types)

    def _close_lmdb(self) -> None:
        self._lmdb.close()
        del self._lmdb
        del self._txn
        self._db_connection_open = False

    def get_keys(self) -> List[bytes]:
        return cast(List[bytes], self._keys)

    def get_shapes(self) -> Any:
        return self._shapes

    def get_types(self) -> Any:
        return self._types

    def read_value_by_key(self, key: bytes) -> Any:
        if not self._db_connection_open:
            self._open_lmdb()

        assert key in self._keys
        return pickle.loads(self._txn.get(key))
