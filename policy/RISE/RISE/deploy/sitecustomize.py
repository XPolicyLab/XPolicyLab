"""RISE evaluation client runtime patches.

XPolicyLab's ModelClient defaults to a 30s socket timeout. RISE's first batched
inference can exceed that while PyTorch warms up, so the eval client waits longer
without changing shared XPolicyLab code.

This module only adjusts the TCP socket recv timeout on ModelClient. It does not
patch model inference, observation history, action chunks, or any server-side
policy logic.
"""

from __future__ import annotations

import builtins
import os


_TIMEOUT = float(os.environ.get("RISE_MODEL_CLIENT_TIMEOUT", "600"))
_ORIGINAL_IMPORT = builtins.__import__
_PATCHED = False


def _patch_model_client(module) -> None:
    global _PATCHED
    if _PATCHED:
        return

    model_client_cls = getattr(module, "ModelClient", None)
    if model_client_cls is None or getattr(model_client_cls, "_rise_timeout_patch", False):
        return

    original_init = model_client_cls.__init__

    def patched_init(self, host="localhost", port=9999, timeout=30):
        if timeout == 30:
            timeout = _TIMEOUT
        original_init(self, host=host, port=port, timeout=timeout)

    model_client_cls.__init__ = patched_init
    model_client_cls._rise_timeout_patch = True
    _PATCHED = True


def _rise_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if name in (
        "client_server.tcp.model_client",
        "XPolicyLab.client_server.tcp.model_client",
    ):
        _patch_model_client(module)
    elif name in ("client_server.tcp", "XPolicyLab.client_server.tcp") and fromlist and "model_client" in fromlist:
        model_client_module = getattr(module, "model_client", None)
        if model_client_module is not None:
            _patch_model_client(model_client_module)
    return module


builtins.__import__ = _rise_import
