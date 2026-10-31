"""Messages between the host and the sandboxed worker.

Both ends exchange pickles over a Unix socket pair.  The host only unpickles with
:class:`HostUnpickler`, which admits plain data, numpy arrays and the result types (RoboDojo's
and the other benchmarks'): a cell cannot smuggle code into the host through the RPC channel.
"""

from __future__ import annotations

import io
import pickle
from multiprocessing.connection import Connection
from typing import Any

_ALLOWED = {
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"), ("builtins", "slice"),
    ("builtins", "range"), ("builtins", "bytearray"), ("collections", "OrderedDict"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
}
_RESULT_MODULES = ("hexashell.robot.results", "hexashell.benches.results")


class HostUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        if module == "numpy.dtypes" and name.endswith("DType"):
            return super().find_class(module, name)
        if module in _RESULT_MODULES and name[:1].isupper():
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"the worker may not send {module}.{name}")


def send(conn: Connection, message: Any) -> None:
    conn.send_bytes(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def recv_trusted(conn: Connection) -> Any:
    """Worker side: messages from the host."""
    return pickle.loads(conn.recv_bytes())


def recv_untrusted(conn: Connection) -> Any:
    """Host side: messages from the worker."""
    return HostUnpickler(io.BytesIO(conn.recv_bytes())).load()
