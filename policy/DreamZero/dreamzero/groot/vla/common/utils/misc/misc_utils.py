import codecs
from collections import Counter
import fnmatch
import hashlib
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

from typing_extensions import Literal


def set_os_envs(envs: Optional[Dict[str, Any]] = None):
    """
    Special value __delete__ or None indicates that the ENV_VAR should be removed
    """
    if envs is None:
        envs = {}
    DEL = {None, "__delete__"}
    for k, v in envs.items():
        if v in DEL:
            os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in envs.items() if v not in DEL})


def argmax(L):
    return max(zip(L, range(len(L))))[1]


def _match_patterns_helper(element, patterns):
    for p in patterns:
        if callable(p) and p(element):
            return True
        if fnmatch.fnmatch(element, p):
            return True
    return False


def match_patterns(
    item: str,
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence: Literal["include", "exclude"] = "exclude",
):
    """
    Args:
        include: None to disable `include` filter and delegate to exclude
        precedence: "include" or "exclude"
    """
    assert precedence in ["include", "exclude"]
    if exclude is None:
        exclude = []
    if isinstance(exclude, (str, Callable)):
        exclude = [exclude]
    if isinstance(include, (str, Callable)):
        include = [include]
    if include is None:
        # exclude is the sole veto vote
        return not _match_patterns_helper(item, exclude)

    if precedence == "include":
        return _match_patterns_helper(item, include)
    else:
        if _match_patterns_helper(item, exclude):
            return False
        else:
            return _match_patterns_helper(item, include)


def filter_patterns(
    items: List[str],
    include: Union[str, List[str], Callable, List[Callable], None] = None,
    exclude: Union[str, List[str], Callable, List[Callable], None] = None,
    *,
    precedence: Literal["include", "exclude"] = "exclude",
    ordering: Literal["original", "include"] = "original",
):
    """
    Args:
        ordering: affects the order of items in the returned list. Does not affect the
            content of the returned list.
            - "original": keep the ordering of items in the input list
            - "include": order items by the order of include patterns
    """
    assert ordering in ["original", "include"]
    if include is None or isinstance(include, str) or ordering == "original":
        return [
            x
            for x in items
            if match_patterns(x, include=include, exclude=exclude, precedence=precedence)
        ]
    else:
        items = items.copy()
        ret = []
        for inc in include:
            for i, item in enumerate(items):
                if item is None:
                    continue
                if match_patterns(item, include=inc, exclude=exclude, precedence=precedence):
                    ret.append(item)
                    items[i] = None
        return ret


def getitem_nested(cfg, key: str):
    """
    Recursively get key, if key has '.' in it
    """
    keys = key.split(".")
    for k in keys:
        assert k in cfg, f'{k} in key "{key}" does not exist in config'
        cfg = cfg[k]
    return cfg


def setitem_nested(cfg, key: str, value):
    """
    Recursively get key, if key has '.' in it
    """
    keys = key.split(".")
    for k in keys[:-1]:
        assert k in cfg, f'{k} in key "{key}" does not exist in config'
        cfg = cfg[k]
    cfg[keys[-1]] = value


def getattr_nested(obj, key: str):
    """
    Recursively get attribute
    """
    keys = key.split(".")
    for k in keys:
        assert hasattr(obj, k), f'{k} in attribute "{key}" does not exist'
        obj = getattr(obj, k)
    return obj


def setattr_nested(obj, key: str, value):
    """
    Recursively set attribute
    """
    keys = key.split(".")
    for k in keys[:-1]:
        assert hasattr(obj, k), f'{k} in attribute "{key}" does not exist'
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)


class PeriodicEvent:
    """
    triggers every period
    """

    def __init__(self, period: int, initial_value=0):
        self._period = period
        assert self._period >= 1
        self._last_threshold = initial_value
        self._last_value = initial_value
        self._trigger_counts = 0

    def __call__(self, new_value=None, increment=None):
        assert bool(new_value is None) != bool(increment is None), (
            "you must specify one and only one of new_value or increment, " "but not both"
        )
        d = self._period
        if new_value is None:
            new_value = self._last_value + increment
        assert new_value >= self._last_value, (
            f"value must be monotonically increasing. "
            f"Current value {new_value} < last value {self._last_value}"
        )
        self._last_value = new_value
        if new_value - self._last_threshold >= d:
            self._last_threshold += (new_value - self._last_threshold) // d * d
            self._trigger_counts += 1
            return True
        else:
            return False

    @property
    def trigger_counts(self):
        return self._trigger_counts

    @property
    def current_value(self):
        return self._last_value


class Once:
    def __init__(self):
        self._triggered = False

    def __call__(self):
        if not self._triggered:
            self._triggered = True
            return True
        else:
            return False

    def __bool__(self):
        raise RuntimeError("`Once` objects should be used by calling ()")


_GLOBAL_ONCE_SET = set()
_GLOBAL_NTIMES_COUNTER = Counter()


def global_once(name):
    """
    Try this to automate the name:
    https://gist.github.com/techtonik/2151727#gistcomment-2333747
    """
    if name in _GLOBAL_ONCE_SET:
        return False
    else:
        _GLOBAL_ONCE_SET.add(name)
        return True


def global_n_times(name, n: int):
    """
    Triggers N times
    """
    assert n >= 1
    if _GLOBAL_NTIMES_COUNTER[name] < n:
        _GLOBAL_NTIMES_COUNTER[name] += 1
        return True
    else:
        return False


class Every:
    def __init__(self, n: int, on_first: bool = False):
        assert n > 0
        self._i = 0 if on_first else 1
        self._n = n

    def __call__(self):
        return self._i % self._n == 0

    def __bool__(self):
        raise RuntimeError("`Every` objects should be used by calling ()")


def encode_base64(obj) -> str:
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def decode_base64(s: str):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
