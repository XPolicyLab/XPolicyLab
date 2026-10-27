# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# This file is taken from https://github.com/NVIDIA/NeMo-Curator, which is adapted from cuML's safe_imports module:
# https://github.com/rapidsai/cuml/blob/e93166ea0dddfa8ef2f68c6335012af4420bc8ac/python/cuml/internals/safe_imports.py


import importlib
import logging
import traceback

import torch
from packaging.version import Version as PkgVersion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

GPU_INSTALL_STRING = (
    """Install GPU packages via `pip install --extra-index-url """
    """https://pypi.nvidia.com nemo-curator[cuda12x]`
or use `pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"` if installing from source"""
)
MISSING_TRITON_MSG = "triton is not installed. Please install it with `pip install triton`."
MISSING_QWEN_VL_UTILS_MSG = "qwen_vl_utils is not installed. Please install it with `pip install qwen-vl-utils`."
MISSING_CUT_CROSS_ENTROPY_MSG = (
    "cut_cross_entropy is not installed. Please install it with `pip install cut-cross-entropy`."
)
MISSING_TORCHAO_MSG = "torchao is not installed. Please install it with `pip install torchao`."


class UnavailableError(Exception):
    """Error thrown if a symbol is unavailable due to an issue importing it."""


def null_decorator(*args, **kwargs):
    if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
        return args[0]
    else:

        def inner(func):
            return func

        return inner


class UnavailableMeta(type):  # noqa D105
    """A metaclass for generating placeholder objects for unavailable symbols."""

    def __new__(meta, name, bases, dct):  # noqa D105
        if dct.get("_msg", None) is None:
            dct["_msg"] = f"{name} could not be imported"
        name = f"MISSING{name}"
        return super(UnavailableMeta, meta).__new__(meta, name, bases, dct)

    def __call__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __getattr__(cls, name):  # noqa D105
        raise UnavailableError(cls._msg)

    def __eq__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __lt__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __gt__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __le__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ge__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ne__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __abs__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __add__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __radd__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __iadd__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __floordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rfloordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ifloordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __lshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rlshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __mul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rmul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __imul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ilshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __pow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rpow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ipow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rrshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __irshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __sub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rsub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __isub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __truediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rtruediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __itruediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __divmod__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rdivmod__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __neg__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __invert__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __hash__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __index__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __iter__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __delitem__(cls, name):  # noqa D105
        raise UnavailableError(cls._msg)

    def __setitem__(cls, name, value):  # noqa D105
        raise UnavailableError(cls._msg)

    def __enter__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __get__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __delete__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __len__(cls):  # noqa D105
        raise UnavailableError(cls._msg)


def is_unavailable(obj):
    return type(obj) is UnavailableMeta


def safe_import(module, *, msg=None, alt=None):
    try:
        return True, importlib.import_module(module)
    except ImportError:
        exception_text = traceback.format_exc()
        logger.debug(f"Import of {module} failed with: {exception_text}")
    except Exception:
        exception_text = traceback.format_exc()
        raise
    if msg is None:
        msg = f"{module} could not be imported"
    if alt is None:
        return False, UnavailableMeta(module.rsplit(".")[-1], (), {"_msg": msg})
    else:
        return False, alt


def safe_import_from(module, symbol, *, msg=None, alt=None, fallback_module=None):
    try:
        imported_module = importlib.import_module(module)
        return True, getattr(imported_module, symbol)
    except ImportError:
        exception_text = traceback.format_exc()
        logger.debug(f"Import of {module} failed with: {exception_text}")
    except AttributeError:
        if fallback_module is not None:
            return safe_import_from(fallback_module, symbol, msg=msg, alt=alt, fallback_module=None)
        exception_text = traceback.format_exc()
        logger.info(f"Import of {symbol} from {module} failed with: {exception_text}")
    except Exception:
        exception_text = traceback.format_exc()
        raise
    if msg is None:
        msg = f"{module}.{symbol} could not be imported"
    if alt is None:
        return False, UnavailableMeta(symbol, (), {"_msg": msg})
    else:
        return False, alt


def gpu_only_import(module, *, alt=None):
    return safe_import(
        module,
        msg=f"{module} is not enabled in non GPU-enabled installations or environemnts. {GPU_INSTALL_STRING}",
        alt=alt,
    )


def gpu_only_import_from(module, symbol, *, alt=None):
    return safe_import_from(
        module,
        symbol,
        msg=f"{module}.{symbol} is not enabled in non GPU-enabled installations or environments. {GPU_INSTALL_STRING}",
        alt=alt,
    )


def safe_import_te():
    msg = "transformer_engine could not be imported (e.g. CUDA symbol mismatch or TE torch lib not found)."
    try:
        import transformer_engine as te
    except ImportError:
        logger.debug("safe_import_te: import transformer_engine failed (e.g. ImportError from torch)", exc_info=True)
        return False, UnavailableMeta("transformer_engine", (), {"_msg": msg})
    except FileNotFoundError:
        logger.debug("safe_import_te: import transformer_engine failed (TE torch lib not found)", exc_info=True)
        return False, UnavailableMeta("transformer_engine", (), {"_msg": msg})

    try:
        pass
    except ImportError:
        logger.debug("safe_import_te: import transformer_engine.pytorch failed", exc_info=True)
        return False, UnavailableMeta("transformer_engine", (), {"_msg": msg})
    except FileNotFoundError:
        logger.debug("safe_import_te: transformer_engine.pytorch failed (TE torch lib not found)", exc_info=True)
        return False, UnavailableMeta("transformer_engine", (), {"_msg": msg})

    return True, te


def get_torch_version():
    try:
        _torch_version = PkgVersion(torch.__version__)
    except Exception:
        _torch_version = PkgVersion("0.0.0")
    return _torch_version


def is_torch_min_version(version, check_equality=True):
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)


def get_te_version():
    try:
        import transformer_engine as te

        if hasattr(te, "__version__"):
            _version = str(te.__version__)
        else:
            from importlib.metadata import version

            _version = version("transformer-engine")
    except ImportError:
        _version = "0.0.0"
    return PkgVersion(_version)


def is_te_min_version(version, check_equality=True):
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)


def get_transformers_version():
    try:
        import transformers

        if hasattr(transformers, "__version__"):
            _version = str(transformers.__version__)
        else:
            from importlib.metadata import version

            _version = version("transformers")
    except ImportError:
        _version = "0.0.0"
    return PkgVersion(_version)


def is_transformers_min_version(version, check_equality=True):
    if check_equality:
        return get_transformers_version() >= PkgVersion(version)
    return get_transformers_version() > PkgVersion(version)


def get_check_model_inputs_decorator():
    """
    Get the appropriate check_model_inputs decorator based on transformers version.

    In transformers >= 4.57.3, check_model_inputs became a function that returns a decorator.
    In older versions, it was directly a decorator.
    In transformers >= 5.2.0, check_model_inputs was removed and split into
    ``merge_with_config_defaults`` and ``capture_outputs``.

    Returns:
        Decorator function to validate model inputs.
    """
    if is_transformers_min_version("5.2.0"):
        try:
            from transformers.utils.generic import merge_with_config_defaults
            from transformers.utils.output_capturing import capture_outputs

            def _combined_decorator(func):
                return merge_with_config_defaults(capture_outputs(func))

            return _combined_decorator
        except ImportError:
            pass

    try:
        from transformers.utils.generic import check_model_inputs

        if is_transformers_min_version("4.57.3"):
            try:
                return check_model_inputs()
            except TypeError:
                return check_model_inputs
        else:
            return check_model_inputs
    except ImportError:
        pass

    return null_decorator
