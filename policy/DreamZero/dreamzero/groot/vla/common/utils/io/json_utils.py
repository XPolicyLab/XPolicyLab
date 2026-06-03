"""
JSON, YAML, and python config file utilities
"""

from io import StringIO
import json
import os.path as path

import yaml

from ..misc.functional_utils import make_recursive_func
from .file_utils import f_join

__all__ = [
    "json_load",
    "json_loads",
    "jsonl_load",
    "yaml_load",
    "yaml_loads",
    "json_dump",
    "json_dumps",
    "jsonl_dump",
    "yaml_dump",
    "yaml_dumps",
    "json_or_yaml_load",
    "json_or_yaml_dump",
    "Jsonl",
    # ---------------- Aliases -----------------
    "load_json",
    "loads_json",
    "load_jsonl",
    "load_yaml",
    "loads_yaml",
    "dump_json",
    "dumps_json",
    "dump_jsonl",
    "dump_yaml",
    "dumps_yaml",
    "load_json_or_yaml",
    "dump_json_or_yaml",
]

from typing import Dict, List

from typing_extensions import Literal


def json_load(*file_path, **kwargs):
    file_path = f_join(file_path)
    with open(file_path, "r") as fp:
        return json.load(fp, **kwargs)


def json_loads(string, **kwargs):
    return json.loads(string, **kwargs)


def jsonl_load(*file_path, **kwargs):
    file_path = f_join(file_path)
    data = []
    for line in open(file_path):
        data.append(json.loads(line, **kwargs))
    return data


@make_recursive_func
def any_to_primitive(x):
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for any_to_primitive")
    import numpy as np

    if isinstance(x, (np.ndarray, np.number, torch.Tensor)):
        return x.tolist()
    else:
        return x


def json_dump(data, *file_path, convert_to_primitive=False, **kwargs):
    if convert_to_primitive:
        data = any_to_primitive(data)
    file_path = f_join(file_path)
    with open(file_path, "w") as fp:
        json.dump(data, fp, **kwargs)


def json_dumps(data, convert_to_primitive=False, **kwargs):
    """
    Returns: string
    """
    if convert_to_primitive:
        data = any_to_primitive(data)
    return json.dumps(data, **kwargs)


def jsonl_dump(data, *file_path):
    from .file_utils import is_sequence

    assert is_sequence(data)
    data = any_to_primitive(data)
    file_path = f_join(file_path)
    with open(file_path, "w") as fp:
        for line in data:
            print(json.dumps(line), file=fp, flush=True)


def yaml_load(*file_path, loader=yaml.safe_load, **kwargs):
    file_path = f_join(file_path)
    with open(file_path, "r") as fp:
        return loader(fp, **kwargs)


def yaml_loads(string, *, loader=yaml.safe_load, **kwargs):
    return loader(string, **kwargs)


def yaml_dump(data, *file_path, dumper=yaml.safe_dump, convert_to_primitive=False, **kwargs):
    if convert_to_primitive:
        data = any_to_primitive(data)
    file_path = f_join(file_path)
    indent = kwargs.pop("indent", 2)
    default_flow_style = kwargs.pop("default_flow_style", False)
    sort_keys = kwargs.pop("sort_keys", False)  # preserves original dict order
    with open(file_path, "w") as fp:
        dumper(
            data,
            stream=fp,
            indent=indent,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            **kwargs,
        )


def yaml_dumps(data, *, dumper=yaml.safe_dump, convert_to_primitive=False, **kwargs):
    "Returns: string"
    if convert_to_primitive:
        data = any_to_primitive(data)
    stream = StringIO()
    indent = kwargs.pop("indent", 2)
    default_flow_style = kwargs.pop("default_flow_style", False)
    sort_keys = kwargs.pop("sort_keys", False)  # preserves original dict order
    dumper(
        data,
        stream,
        indent=indent,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        **kwargs,
    )
    return stream.getvalue()


# ==================== auto-recognize extension ====================
def json_or_yaml_load(*file_path, **loader_kwargs):
    """
    Args:
        file_path: JSON or YAML loader depends on the file extension

    Raises:
        IOError: if extension is not ".json", ".yml", or ".yaml"
    """
    file_path = str(f_join(file_path))
    if file_path.endswith(".json"):
        return json_load(file_path, **loader_kwargs)
    elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
        return yaml_load(file_path, **loader_kwargs)
    else:
        raise IOError(
            f'unknown file extension: "{file_path}", '
            f'loader supports only ".json", ".yml", ".yaml"'
        )


def json_or_yaml_dump(data, *file_path, **dumper_kwargs):
    """
    Args:
        file_path: JSON or YAML loader depends on the file extension

    Raises:
        IOError: if extension is not ".json", ".yml", or ".yaml"
    """
    file_path = str(f_join(file_path))
    if file_path.endswith(".json"):
        return json_dump(data, file_path, **dumper_kwargs)
    elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
        return yaml_dump(data, file_path, **dumper_kwargs)
    else:
        raise IOError(
            f'unknown file extension: "{file_path}", '
            f'dumper supports only ".json", ".yml", ".yaml"'
        )


# ---------------- Aliases -----------------
# add aliases where verb goes first, json_load -> load_json
load_json = json_load
load_yaml = yaml_load
load_jsonl = jsonl_load
loads_json = json_loads
loads_yaml = yaml_loads
dump_json = json_dump
dump_jsonl = jsonl_dump
dump_yaml = yaml_dump
dumps_json = json_dumps
dumps_yaml = yaml_dumps
load_json_or_yaml = json_or_yaml_load
dump_json_or_yaml = json_or_yaml_dump

# ==================== Jsonl ====================


class Jsonl:
    """
    Both reader and writer, as if everything's in-memory
    """

    def __init__(self, *file_path, mode: Literal["r", "w", "a"] = "a"):
        """
        Args:
            mode:
            - 'r': file must already exists
            - 'w': overwrite the file regardless of whether it exists or not
            - 'a': create a new file if doesn't exist, or append to an existing file
        """
        assert mode in "rwa"
        self._file_path = str(f_join(file_path))
        self._mode = mode
        if mode == "r":
            assert path.exists(self._file_path)
            self._fp = None
        else:
            self._fp = open(self._file_path, mode)
        if path.exists(self._file_path) and mode != "w":
            self.data = jsonl_load(self._file_path)
        else:
            self.data = []

    def append(self, data: Dict):
        if self._mode == "r":
            raise RuntimeError("Jsonl read mode cannot call append()")
        self.data.append(data)
        print(json_dumps(data), file=self._fp, flush=True)

    def extend(self, data_list: List[Dict]):
        for data in data_list:
            self.append(data)

    def close(self):
        if self._fp is not None:
            self._fp.close()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __bool__(self):
        return bool(self.data)
