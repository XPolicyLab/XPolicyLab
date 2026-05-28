"""Register Dexbotic model builders into RLinf ``register_model`` / ``_MODEL_REGISTRY``.

Driver process: call ``register_all()`` from entry scripts (or rely on ``register()`` via
``RLINF_EXT_MODULE`` — see below).

Ray worker processes: RLinf loads this module from env ``RLINF_EXT_MODULE`` and calls
``register()`` once per worker (``Worker._load_user_extensions``). ``register()`` delegates
to ``register_all()`` so workers see the same custom ``model_type`` keys as the driver.

Example launch (from repo root, with ``EMBODIED_PATH`` pointing at RLinf
``examples/embodiment``):

    export RLINF_EXT_MODULE=dexbotic.rl.rlinf_registry
    export EMBODIED_PATH=/path/to/RLinf/examples/embodiment
    python -m dexbotic.rl.model_rl_libero_pi0

``dexbotic.rl._embodied_cli`` also sets ``RLINF_EXT_MODULE`` via ``setdefault`` so the
export is optional when using those entrypoints.
"""

from __future__ import annotations

from rlinf.models import register_model


def _get_pi0_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_pi0_policy import get_model

    return get_model(cfg, torch_dtype)


def _get_dm0_model(cfg, torch_dtype=None):
    from dexbotic.rl.rlinf_bridge.dexbotic_dm0_policy import get_model

    return get_model(cfg, torch_dtype)


def register_all() -> None:
    """Register Dexbotic ``model_type`` strings with RLinf (``register_model``).

    Uses ``force=True`` so repeated calls replace cleanly and Dexbotic loaders override
    the built-in ``dexbotic_pi`` entry when needed.
    """
    register_model("dexbotic_pi0", _get_pi0_model, category="embodied", force=True)
    register_model("dexbotic_pi", _get_pi0_model, category="embodied", force=True)
    register_model("dexbotic_dm0", _get_dm0_model, category="embodied", force=True)


def register() -> None:
    """RLinf ``RLINF_EXT_MODULE`` hook: invoked once per Worker process."""
    register_all()
