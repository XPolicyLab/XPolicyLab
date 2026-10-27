# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import ast
import importlib
import importlib.util
import inspect
import os
import pprint
import re
import sys
import types
from copy import deepcopy
from pathlib import Path

from typing import Any, Mapping

import yaml

ALLOWED_IMPORT_PREFIXES = ("awomo", "torch", "transformers", "torchdata", "torchao", "liger_kernel")

SAFE_BASE_DIR = Path(__file__).resolve().parents[2]

ENABLE_USER_MODULES = os.environ.get("NEMO_ENABLE_USER_MODULES", "").lower() in ("1", "true", "yes")

SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "apikey",
    "api_key",
    "authorization",
    "auth",
)


def set_enable_user_modules(allow: bool) -> None:
    global ENABLE_USER_MODULES
    ENABLE_USER_MODULES = bool(allow)


def _is_safe_path(p: Path) -> bool:
    rp = p.resolve()
    try:
        return rp.is_relative_to(SAFE_BASE_DIR)
    except AttributeError:
        try:
            return os.path.commonpath([str(rp), str(SAFE_BASE_DIR.resolve())]) == str(SAFE_BASE_DIR.resolve())
        except ValueError:
            return False


def _is_allowed_module(module_name: str) -> bool:
    if ENABLE_USER_MODULES:
        return True

    top_level = module_name.split(".")[0]

    if top_level in sys.modules:
        return True

    try:
        spec = importlib.util.find_spec(top_level)
    except Exception:
        spec = None
    if spec is not None:
        return True

    return any(top_level == pref or top_level.startswith(pref + ".") for pref in ALLOWED_IMPORT_PREFIXES)


def _is_safe_attr(name: str) -> bool:
    return not (name.startswith("_") or "__" in name)


def _redact(obj: Any) -> Any:
    def needs_redact(k: str) -> bool:
        lk = str(k).lower()
        return any(s in lk for s in SENSITIVE_KEY_SUBSTRINGS)

    if isinstance(obj, Mapping):
        return {k: ("******" if needs_redact(k) else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def translate_value(v: Any) -> Any:
    if not isinstance(v, str):
        return v

    special_symbols = {
        "none": None,
        "None": None,
        "true": True,
        "True": True,
        "false": False,
        "False": False,
    }
    if v in special_symbols:
        return special_symbols[v]

    if len(v) > 1000:
        return v

    try:
        return ast.literal_eval(v)
    except Exception:
        return v


class _OrigValueStr(str):
    """String that keeps its original (unresolved) value for safe printing."""

    _orig_value: str
    _no_env_resolve: bool

    def __new__(cls, value: str, orig_value: str) -> "_OrigValueStr":
        obj = super().__new__(cls, value)
        obj._orig_value = orig_value
        obj._no_env_resolve = True
        return obj


def resolve_yaml_env_vars(obj: Any) -> Any:

    def _resolve_in_str(value: str) -> str:
        if hasattr(value, "_no_env_resolve"):
            return value
        if "$" not in value:
            return value

        def _get_env(var_name: str, default: str | None, token: str) -> str:
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default
            raise KeyError(f"Environment variable '{var_name}' is not set (required by '{token}').")

        braced_pattern = re.compile(r"\$\{([^}]+)\}")

        def _braced_repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            if expr.startswith("oc.env:"):
                expr = expr[len("oc.env:") :].strip()

            if "," in expr:
                var_name, default = expr.split(",", 1)
                var_name = var_name.strip()
                default = default.strip()
            else:
                var_name, default = expr, None

            return _get_env(var_name, default, match.group(0))

        value = braced_pattern.sub(_braced_repl, value)

        dollar_pattern = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)")

        def _dollar_repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return _get_env(var_name, None, match.group(0))

        return dollar_pattern.sub(_dollar_repl, value)

    if isinstance(obj, dict):
        return {k: resolve_yaml_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_yaml_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_in_str(obj)
    return obj


def load_module_from_file(file_path: str | Path) -> types.ModuleType:
    p = Path(file_path)
    if p.suffix != ".py":
        raise ImportError(f"Refusing to load non-Python file as module: {p}")

    unique_part = str(abs(hash(str(p.resolve()))) % (10**8))
    name = f"cfgmod_{unique_part}_{p.stem}"
    spec = importlib.util.spec_from_file_location(name, str(p.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {p}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _resolve_target(dotted_path: str) -> Any:
    if not isinstance(dotted_path, str):
        return dotted_path

    if ":" in dotted_path:
        file_part, attr = dotted_path.split(":", 1)
        p = Path(file_part)
        if p.suffix != ".py":
            raise AssertionError("Left side of ':' must be a .py file")
        if not p.exists():
            raise AssertionError(f"Python script does not exist: {file_part}")
        module = load_module_from_file(str(p.resolve()))
        if not _is_safe_attr(attr):
            raise ImportError(
                "Access to private or dunder attributes is disabled by default. "
                "To allow out-of-tree code, set NEMO_ENABLE_USER_MODULES=1 or call set_enable_user_modules(True)."
            )
        return getattr(module, attr)

    parts = dotted_path.split(".")

    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        remainder = parts[i:]

        if not _is_allowed_module(module_name):
            continue

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        obj = module
        for name in remainder:
            if not _is_safe_attr(name) and not ENABLE_USER_MODULES:
                raise ImportError(
                    "Access to private or dunder attributes is disabled by default. "
                    "To allow out-of-tree code, set NEMO_ENABLE_USER_MODULES=1 or call set_enable_user_modules(True)."
                )
            try:
                obj = getattr(obj, name)
            except AttributeError:
                raise ImportError(
                    f"Module '{module_name}' loaded, but cannot resolve attribute '{'.'.join(remainder)}' in '{dotted_path}'"
                )
        return obj

    raise ImportError(f"Cannot resolve target (blocked or not found): {dotted_path}")


class ConfigNode:
    """A configuration node that wraps a dictionary (or parts of it) from a YAML file."""

    def __init__(self, d: dict[str, Any], raise_on_missing_attr: bool = True) -> None:
        self._raw_config = deepcopy(d)
        self._original_strings: dict[str, str] = {}
        self.__dict__.update({k: self._wrap(k, v) for k, v in d.items()})
        self.raise_on_missing_attr = raise_on_missing_attr

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        try:
            return self.__dict__[key]
        except KeyError:
            if self.__dict__.get("raise_on_missing_attr", True):
                raise AttributeError(key)
            return None

    def _wrap(self, k: str, v: Any) -> Any:
        """Wrap a configuration value based on its type.

        Args:
            k (str): The key corresponding to the value.
            v: The value to be wrapped.

        Returns:
            The wrapped value.
        """
        if isinstance(v, dict):
            return ConfigNode(v)
        elif isinstance(v, list):
            return [self._wrap("", i) for i in v]
        elif k.endswith("_fn"):
            if isinstance(v, str):
                self._original_strings[k] = v
            return _resolve_target(v)
        elif k == "_target_":
            if isinstance(v, str):
                self._original_strings[k] = v
            return _resolve_target(v)
        else:
            if isinstance(v, str) and "$" in v:
                resolved = resolve_yaml_env_vars(v)
                translated = translate_value(resolved)
                if isinstance(translated, str) and resolved != v:
                    return _OrigValueStr(translated, v)
                return translated
            return translate_value(v)

    @property
    def raw_config(self) -> dict[str, Any]:
        return self._raw_config

    def instantiate_path(self, dotted_path: str, default: Any = None, *args: Any, **kwargs: Any) -> Any:
        item = self.get(dotted_path, default)
        if item is default:
            return default
        return item.instantiate(*args, **kwargs)

    def instantiate(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "_target_"):
            raise AttributeError("No _target_ found to instantiate")

        func = _resolve_target(self._target_)

        config_kwargs = {}
        for k, v in self.__dict__.items():
            if k in ("_target_", "raise_on_missing_attr", "_raw_config", "_original_strings"):
                continue
            if k in kwargs:
                continue
            if k.endswith("_fn"):
                config_kwargs[k] = v
            else:
                config_kwargs[k] = self._instantiate_value(v)

        config_kwargs = resolve_yaml_env_vars(config_kwargs)

        config_kwargs.update(kwargs)

        import traceback

        try:
            return func(*args, **config_kwargs)
        except Exception as e:
            sig = inspect.signature(func)
            safe_kwargs = _redact(config_kwargs)
            print(
                "Instantiation failed for `{}`\n"
                "Accepted signature : {}\n"
                "Positional args    : {}\n"
                "Keyword args       : {}\n"
                "Exception          : {}\n".format(
                    getattr(func, "__name__", str(func)),
                    sig,
                    args,
                    pprint.pformat(safe_kwargs, compact=True, indent=4),
                    e,
                ),
                file=sys.stderr,
            )
            print(traceback.format_exc())
            raise e

    def _instantiate_value(self, v: Any) -> Any:
        if isinstance(v, ConfigNode) and hasattr(v, "_target_"):
            return v.instantiate()
        elif isinstance(v, ConfigNode):
            return resolve_yaml_env_vars(v.to_dict())
        elif isinstance(v, list):
            return [self._instantiate_value(i) for i in v]
        else:
            return translate_value(resolve_yaml_env_vars(v))

    def to_dict(self) -> dict[str, Any]:
        return {
            k: self._unwrap(v)
            for k, v in self.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }

    def _to_dotted_path(self, obj: Any) -> str:
        try:
            import inspect as _inspect

            if _inspect.ismethod(obj):
                owner = getattr(obj, "__self__", None)
                if _inspect.isclass(owner):
                    method_name = getattr(obj, "__name__", "unknown")
                    module_name = getattr(owner, "__module__", None) or ""
                    class_name = getattr(owner, "__name__", "UnknownClass")
                    if class_name.startswith("NeMoAutoModel"):
                        module_name = "awomo"
                    dotted = f"{module_name}.{class_name}.{method_name}".lstrip(".")
                else:
                    module_name = getattr(obj, "__module__", None) or ""
                    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "unknown"))
                    dotted = f"{module_name}.{qualname}".lstrip(".")
            elif _inspect.isfunction(obj):
                module_name = getattr(obj, "__module__", None) or ""
                qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "unknown"))
                dotted = f"{module_name}.{qualname}".lstrip(".")
            elif _inspect.isclass(obj):
                module_name = getattr(obj, "__module__", None) or ""
                class_name = getattr(obj, "__name__", "UnknownClass")
                dotted = f"{module_name}.{class_name}".lstrip(".")
            else:
                module_name = getattr(obj, "__module__", None) or ""
                qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
                dotted = f"{module_name}.{qualname}".lstrip(".")
        except Exception:
            return repr(obj)
        return dotted

    def to_yaml_dict(
        self, *, resolve_env: bool = False, redact_sensitive: bool = False, use_orig_values: bool = False
    ) -> dict[str, Any]:

        def _convert(key: str | None, value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return value.to_yaml_dict(
                    resolve_env=resolve_env,
                    redact_sensitive=redact_sensitive,
                    use_orig_values=use_orig_values,
                )
            if isinstance(value, list):
                return [_convert(None, v) for v in value]
            if isinstance(value, dict):
                return {k: _convert(k, v) for k, v in value.items()}
            orig_strings = getattr(self, "_original_strings", {})
            if use_orig_values and key in orig_strings:
                return orig_strings[key]
            is_target_like = key == "_target_" or (isinstance(key, str) and key.endswith("_fn")) or key == "collate_fn"
            try:
                import inspect as _inspect

                if is_target_like and (callable(value) or _inspect.ismethod(value) or _inspect.isclass(value)):
                    return self._to_dotted_path(value)
                if callable(value) or _inspect.ismethod(value) or _inspect.isclass(value):
                    return self._to_dotted_path(value)
            except Exception:
                pass
            if use_orig_values and hasattr(value, "_orig_value"):
                return getattr(value, "_orig_value")
            return value

        out = {
            k: _convert(k, v)
            for k, v in self.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }
        if resolve_env:
            out = resolve_yaml_env_vars(out)
        if redact_sensitive:
            out = _redact(out)
        return out

    def _unwrap(self, v: Any) -> Any:
        if isinstance(v, ConfigNode):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._unwrap(i) for i in v]
        else:
            return v

    def get_as_string(self, key: str, default: str | None = None) -> str:
        if key in self._original_strings:
            return self._original_strings[key]
        if default is not None:
            return default
        raise KeyError(f"Key {key} not found")

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        current = self
        for p in parts:
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return default
            elif isinstance(current, list):
                try:
                    idx = int(p)
                    current = current[idx]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return current

    def set_by_dotted(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self
        for p in parts[:-1]:
            if p not in node.__dict__ or not isinstance(node.__dict__[p], ConfigNode):
                node.__dict__[p] = ConfigNode({})
            node = node.__dict__[p]
        node.__dict__[parts[-1]] = node._wrap(parts[-1], value)

    def _format(self, level: int = 0, *, use_orig_values: bool = True) -> str:
        indent = "  " * level
        lines = [
            f"{indent}{key}: {self._repr_value(value, level, use_orig_values=use_orig_values)}"
            for key, value in self.__dict__.items()
            if key not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        ]
        return "\n".join(lines) + f"\n{indent}"

    def __repr__(self) -> str:
        return self._format(level=0, use_orig_values=True)

    def _repr_value(self, value: Any, level: int, *, use_orig_values: bool = True) -> str:
        if isinstance(value, ConfigNode):
            return value._format(level + 1, use_orig_values=use_orig_values)
        elif isinstance(value, list):
            return (
                "[\n"
                + "\n".join(
                    [
                        f"{'  ' * (level + 1)}{self._repr_value(i, level + 1, use_orig_values=use_orig_values)}"
                        for i in value
                    ]
                )
                + f"\n{'  ' * level}]"
            )
        else:
            if use_orig_values and hasattr(value, "_orig_value"):
                return repr(getattr(value, "_orig_value"))
            return repr(value)

    def __str__(self) -> str:
        return self._format(level=0, use_orig_values=True)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        parts = key.split(".")
        current: Any = self
        for p in parts:
            if isinstance(current, ConfigNode):
                if p in current.__dict__:
                    current = current.__dict__[p]
                else:
                    return False
        return bool(current != self)


def config_to_yaml_str(cfg_obj: Any, *, use_orig_values: bool = True) -> str:
    if cfg_obj is None:
        return ""
    if hasattr(cfg_obj, "to_yaml_dict"):
        cfg_dict = cfg_obj.to_yaml_dict(use_orig_values=use_orig_values)
    elif hasattr(cfg_obj, "to_dict"):
        cfg_dict = cfg_obj.to_dict()
    else:
        cfg_dict = dict(cfg_obj)
    result: str = yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False).strip()
    return result


def load_yaml_config(path: str | Path) -> "ConfigNode":
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)
