# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import pickle
from typing import Union

import lmdb
from timeout_decorator import TimeoutError, timeout

logger = logging.getLogger(__name__)


class Lmdb(object):
    """Abstact class of LMDB, which include all operators.

    Args:
        uri: Path to lmdb file.
        writable: Writable flag for opening LMDB.
        commit_step: The step for commit.
        map_size:
            Maximum size database may grow to, used to size the memory mapping.
            If map_size is None, map_size will set to 10M while reading,
            set to 1T while writing.
        kwargs: Kwargs for open lmdb file.
    """

    def __init__(
        self,
        uri: str,
        writable: bool = True,
        commit_step: int = 1,
        reset_step: int = -1,
        map_size: int = None,
        encoding_mode: str = "utf-8",
        **kwargs,
    ):
        self.uri = uri
        self.writable = writable
        self.kwargs = kwargs
        self.kwargs["map_size"] = map_size
        # default lmdb settings

        self.kwargs["meminit"] = self.kwargs.get("meminit", False)
        self.kwargs["map_async"] = self.kwargs.get("map_async", True)
        self.kwargs["sync"] = self.kwargs.get("sync", False)
        if not writable:
            self.kwargs["readonly"] = True
            self.kwargs["lock"] = False
            # set map_size to 10M while reading.
            if self.kwargs.get("map_size") is None:
                self.kwargs["map_size"] = 10485760
        else:
            # set map_size to 1T while writing.
            if self.kwargs.get("map_size") is None:
                self.kwargs["map_size"] = 1024**4
        # LMDB env
        self.env = None
        self.txn = None
        self.open()
        if not self.writable:
            self._create_txn()
        # pack settings
        self.commit_step = commit_step
        self.reset_step = reset_step
        self.encoding_mode = encoding_mode
        self.put_idx = 0
        self.get_idx = 0

    def read(self, idx: Union[int, str]) -> bytes:
        """Read data by idx."""
        idx = "{}".format(idx).encode(self.encoding_mode)
        try:
            return self.get(idx)
        except TimeoutError as exception:
            logger.error(
                f"Time out when reading data with index of "
                f"{idx} from {self.uri}"
            )
            raise exception

    @timeout(seconds=1800)
    def get(self, idx: Union[int, str]) -> bytes:
        if not isinstance(idx, bytes):
            idx = "{}".format(idx).encode(self.encoding_mode)
        if self.txn is None:
            self._create_txn()
        data = self.txn.get(idx)
        if self.reset_step > 0:
            self.get_idx += 1
            if self.get_idx % self.reset_step == 0:
                self.get_idx = 0
                self.reset()
        if data is not None:
            return pickle.loads(data)
        return None

    def __getitem__(self, idx: Union[int, str]) -> bytes:
        return self.get(idx)

    def write(self, idx: Union[int, str], record: bytes, commit=False):
        """Write data into lmdb file."""
        if self.env is None:
            self.open()
        if self.txn is None:
            self._create_txn()
        record = pickle.dumps(record, protocol=4)
        self.txn.put("{}".format(idx).encode(self.encoding_mode), record)
        self.put_idx += 1
        if (self.put_idx % self.commit_step == 0) or commit:
            self.txn.commit()
            self.txn = self.env.begin(write=self.writable)

    @timeout(seconds=1800)
    def open_lmdb(self):
        return lmdb.open(self.uri, **self.kwargs)

    def open(self):
        """Open lmdb file."""
        if self.env is None:
            try:
                self.env = self.open_lmdb()
            except TimeoutError as exception:
                logger.error(f"Time out when opening {self.uri}")
                raise exception

    def _create_txn(self):
        """Create lmdb transaction."""
        if self.env is None:
            self.open()
        if self.txn is None:
            self.txn = self.env.begin(write=self.writable)

    def close(self):
        """Close lmdb file."""
        if self.env is not None:
            if self.writable and self.txn is not None:
                self.txn.commit()
                self.put_idx = 0
                self.env.sync()
            self.env.close()
            self.env = None
            self.txn = None

    def reset(self):
        """Reset open file."""
        if self.env is None and self.txn is None:
            self.open()
        else:
            self.close()
            self.open()
            if not self.writable:
                self._create_txn()

    def keys(self):
        """Get all keys."""
        if self.txn is None:
            self._create_txn()
        try:
            idx = "{}".format("__len__").encode(self.encoding_mode)
            return range(int(self.txn.get(idx)))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            for key, _value in self.txn.cursor():
                keys.append(key.decode(self.encoding_mode))
            return keys

    def __len__(self):
        """Get the length."""
        if self.txn is None:
            self._create_txn()
        try:
            idx = "{}".format("__len__").encode(self.encoding_mode)
            return int(self.txn.get(idx))
        except Exception:
            return self.txn.stat()["entries"]

    def __getstate__(self):
        state = self.__dict__
        self.close()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.open()
