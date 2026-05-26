from copy import deepcopy
import importlib.resources
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import tree

from ..misc.functional_utils import call_once, is_mapping, is_sequence, meta_decorator
from .print_utils import to_scientific_str

_CLASS_REGISTRY = {}  # for instantiation


def resource_file_path(pkg_name, fname) -> str:
    with importlib.resources.path(pkg_name, fname) as p:
        return str(p)


def print_config(cfg: DictConfig):
    print(cfg.pretty(resolve=True))


def is_hydra_initialized():
    return hydra.utils.HydraConfig.initialized()


def hydra_config():
    # https://github.com/facebookresearch/hydra/issues/377
    # HydraConfig() is a singleton
    if is_hydra_initialized():
        return hydra.utils.HydraConfig().cfg.hydra
    else:
        return None


def hydra_override_arg_list() -> list[str]:
    """
    Returns:
        list ["lr=0.2", "batch=64", ...]
    """
    if is_hydra_initialized():
        return hydra_config().overrides.task
    else:
        return []


def hydra_override_name():
    if is_hydra_initialized():
        return hydra_config().job.override_dirname
    else:
        return ""


def hydra_original_dir(*subpaths):
    return os.path.join(hydra.utils.get_original_cwd(), *subpaths)


@call_once(on_second_call="noop")
def register_omegaconf_resolvers():
    import numpy as np

    OmegaConf.register_new_resolver("scientific", lambda v, i=0: to_scientific_str(v, i))
    OmegaConf.register_new_resolver("_optional", lambda v: f"_{v}" if v else "")
    OmegaConf.register_new_resolver("optional_", lambda v: f"{v}_" if v else "")
    OmegaConf.register_new_resolver("_optional_", lambda v: f"_{v}_" if v else "")
    OmegaConf.register_new_resolver("__optional", lambda v: f"__{v}" if v else "")
    OmegaConf.register_new_resolver("optional__", lambda v: f"{v}__" if v else "")
    OmegaConf.register_new_resolver("__optional__", lambda v: f"__{v}__" if v else "")
    OmegaConf.register_new_resolver("iftrue", lambda cond, v_default: cond if cond else v_default)
    OmegaConf.register_new_resolver("ifelse", lambda cond, v1, v2="": v1 if cond else v2)
    OmegaConf.register_new_resolver(
        "ifequal", lambda query, key, v1, v2: v1 if query == key else v2
    )
    OmegaConf.register_new_resolver("intbool", lambda cond: 1 if cond else 0)
    OmegaConf.register_new_resolver("mult", lambda *x: np.prod(x).tolist())
    OmegaConf.register_new_resolver("add", lambda *x: sum(x))
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x // y)

    # try each key until the key exists. Useful for multiple classes that have different
    # names for the same key
    def _try_key(cfg, *keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
        raise KeyError(f"no key in {keys} is valid")

    OmegaConf.register_new_resolver("trykey", _try_key)
    # replace `resnet.gn.ws` -> `resnet_gn_ws`, because omegaconf doesn't support
    # keys with dots. Useful for generating run name with dots
    OmegaConf.register_new_resolver("underscore_to_dots", lambda s: s.replace("_", "."))

    def _no_instantiate(cfg):
        cfg = deepcopy(cfg)
        cfg[_NO_INSTANTIATE] = True
        return cfg

    OmegaConf.register_new_resolver("no_instantiate", _no_instantiate)


# ========================================================
# ================== Instantiation tools  ================
# ========================================================


def register_callable(name, class_type):
    if isinstance(class_type, str):
        class_type, name = name, class_type
    assert callable(class_type)
    _CLASS_REGISTRY[name] = class_type


@meta_decorator
def register_class(cls, alias=None):
    """
    Decorator
    """
    assert callable(cls)
    _CLASS_REGISTRY[cls.__name__] = cls
    if alias:
        assert is_sequence(alias)
        for a in alias:
            _CLASS_REGISTRY[str(a)] = cls
    return cls


def omegaconf_to_dict(cfg, resolve: bool = True, enum_to_str: bool = False):
    """
    Convert arbitrary nested omegaconf objects to primitive containers

    WARNING: cannot use tree lib because it gets confused on DictConfig and ListConfig
    """
    kw = dict(resolve=resolve, enum_to_str=enum_to_str)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, **kw)
    elif is_sequence(cfg):
        return type(cfg)(omegaconf_to_dict(c, **kw) for c in cfg)
    elif is_mapping(cfg):
        return {k: omegaconf_to_dict(c, **kw) for k, c in cfg.items()}
    else:
        return cfg


def omegaconf_save(cfg, *paths: str, resolve: bool = True):
    """
    Save omegaconf to yaml
    """
    from .file_utils import f_join

    OmegaConf.save(cfg, f_join(*paths), resolve=resolve)


def get_class(path):
    """
    First try to find the class in the registry first,
    if it doesn't exist, use importlib to locate it
    """
    if path in _CLASS_REGISTRY:
        return _CLASS_REGISTRY[path]
    else:
        assert "." in path, (
            f"Because {path} is not found in class registry, " f"it must be a full module path"
        )
    try:
        from importlib import import_module

        module_path, _, class_name = path.rpartition(".")
        mod = import_module(module_path)
        try:
            class_type = getattr(mod, class_name)
        except AttributeError:
            raise ImportError("Class {} is not in module {}".format(class_name, module_path))
        return class_type
    except ValueError as e:
        print("Error initializing class " + path, file=sys.stderr)
        raise e


_DELETE_ARG = "__delete__"
_NO_INSTANTIATE = "__no_instantiate__"  # return config as-is
_OMEGA_MISSING = "???"


def _get_instantiate_params(cfg, kwargs=None):
    params = cfg
    f_args, f_kwargs = (), {}
    for k, value in params.items():
        if k in ["cls", "class"]:
            continue
        elif k == "*args":
            assert is_sequence(value), '"*args" value must be a sequence'
            f_args = list(value)
            continue
        if value == _OMEGA_MISSING:
            if kwargs and k in kwargs:
                value = kwargs[k]
            else:
                raise ValueError(f'Missing required keyword arg "{k}" in cfg: {cfg}')
        if value == _DELETE_ARG:
            continue
        else:
            f_kwargs[k] = value
    return f_args, f_kwargs


def _instantiate_single(cfg):
    if is_mapping(cfg) and ("cls" in cfg or "class" in cfg):
        assert bool("cls" in cfg) != bool("class" in cfg), (
            "to instantiate from config, "
            'one and only one of "cls" or "class" key should be provided'
        )
        if _NO_INSTANTIATE in cfg:
            no_instantiate = cfg.pop(_NO_INSTANTIATE)
            if no_instantiate:
                cfg = deepcopy(cfg)
                return cfg
            else:
                return _instantiate_single(cfg)

        cls = cfg.get("class", cfg.get("cls"))
        args, kwargs = _get_instantiate_params(cfg)
        try:
            class_type = get_class(cls)
            return class_type(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error instantiating {cls}: {e}")
    else:
        return None


def instantiate(_cfg_, **kwargs):
    """
    Any dict with "cls" or "class" key is considered instantiable.

    Any key that has the special value "__delete__"
    will not be passed to the constructor

    **kwargs only apply to the top level object if it's a dict, otherwise raise error
    """
    assert OmegaConf.is_config(_cfg_) or isinstance(_cfg_, (list, tuple)) or is_mapping(_cfg_), (
        '"cfg" must be a dict, list, tuple, or OmegaConf config to be instantiated. '
        f"Current its type is {type(_cfg_)}"
    )

    _cfg_ = omegaconf_to_dict(_cfg_, resolve=True)

    if kwargs:
        if is_mapping(_cfg_):
            _cfg_ = _cfg_.copy()
            _cfg_.update(kwargs)
            _cfg_ = {k: v for k, v in _cfg_.items() if v != _DELETE_ARG}
        else:
            raise RuntimeError(
                f"**kwargs specified, but the top-level cfg is not a dict. "
                f"It has type {type(_cfg_)}"
            )

    return tree.traverse(_instantiate_single, _cfg_, top_down=False)
