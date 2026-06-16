"""Spirit_v15 eval client: extend ModelClient socket timeout for slow VLM inference."""

from __future__ import annotations

import builtins
import os

_TIMEOUT = float(os.environ.get("SPIRIT_MODEL_CLIENT_TIMEOUT", "600"))
_ORIGINAL_IMPORT = builtins.__import__
_PATCHED = False


def _patch_model_client(module) -> None:
    global _PATCHED
    if _PATCHED:
        return

    model_client_cls = getattr(module, "ModelClient", None)
    if model_client_cls is None or getattr(model_client_cls, "_spirit_timeout_patch", False):
        return

    original_init = model_client_cls.__init__

    def patched_init(self, host="localhost", port=9999, timeout=30):
        if timeout == 30:
            timeout = _TIMEOUT
        original_init(self, host=host, port=port, timeout=timeout)

    model_client_cls.__init__ = patched_init
    model_client_cls._spirit_timeout_patch = True
    _PATCHED = True


def _spirit_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if name in ("client_server.model_client", "XPolicyLab.client_server.model_client"):
        _patch_model_client(module)
    elif name in ("client_server", "XPolicyLab.client_server") and fromlist and "model_client" in fromlist:
        model_client_module = getattr(module, "model_client", None)
        if model_client_module is not None:
            _patch_model_client(model_client_module)
    return module


builtins.__import__ = _spirit_import
