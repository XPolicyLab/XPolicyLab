"""Safe-loading compatibility from the companion Phoenix-MWA-RTC repository.

This supplies the missing Torch 2.4 inspection/context APIs, never an
unrestricted pickle fallback. It is enabled explicitly by the cache extractor.
"""
from contextlib import contextmanager
import _codecs
import pickletools
import threading
import zipfile

import torch


_LOCK = threading.RLock()


def install_safe_serialization_compat():
    if not hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        torch.serialization.add_safe_globals([_codecs.encode])

        def scan(path):
            from torch import _weights_only_unpickler
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
                if len(names) != 1:
                    raise ValueError("Expected one torch ZIP pickle payload")
                data = archive.read(names[0])
            found = set()
            for opcode, arg, _ in pickletools.genops(data):
                if opcode.name == "GLOBAL":
                    module, name = arg.split(" ", 1)
                    found.add(module + "." + name)
                elif opcode.name in ("STACK_GLOBAL", "EXT1", "EXT2", "EXT4"):
                    raise ValueError("Unsupported pickle global encoding; refusing unsafe load")
            return sorted(found - set(_weights_only_unpickler._get_allowed_globals())
                          - set(_weights_only_unpickler._get_user_allowed_globals()))

        torch.serialization.get_unsafe_globals_in_checkpoint = scan
    if not hasattr(torch.serialization, "safe_globals"):
        @contextmanager
        def safe_globals(values):
            with _LOCK:
                saved = torch.serialization.get_safe_globals()
                torch.serialization.add_safe_globals(values)
                try:
                    yield
                finally:
                    torch.serialization.clear_safe_globals()
                    torch.serialization.add_safe_globals(saved)
        torch.serialization.safe_globals = safe_globals
