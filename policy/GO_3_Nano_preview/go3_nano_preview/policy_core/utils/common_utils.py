import contextlib
import dataclasses
import warnings

import torch
import torch.utils._pytree as pytree

from policy_core.settings import settings

_REGISTERED_DATACLASSES: set[type] = set()


def normalize_robot_type(robot_type: object) -> str:
    if str(robot_type).strip().lower() != 'robotwin':
        raise ValueError('Only robotwin robot type is supported')
    return 'robotwin'


def get_device():
    return torch.device(
        f"cuda:{settings.local_rank}" if torch.cuda.is_available() else "cpu"
    )


def _register_dataclass_as_pytree(cls: type) -> None:
    if cls in _REGISTERED_DATACLASSES:
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*is already registered as pytree node.*",
            category=UserWarning,
        )
        if hasattr(pytree, "register_dataclass"):
            with contextlib.suppress(Exception):
                pytree.register_dataclass(cls)
                _REGISTERED_DATACLASSES.add(cls)
                return

        field_names = [f.name for f in dataclasses.fields(cls)]

        def _flatten(obj, _names=field_names):
            present = [
                (n, getattr(obj, n)) for n in _names if getattr(obj, n) is not None
            ]
            return [v for _, v in present], [n for n, _ in present]

        def _unflatten(values, context, _cls=cls, _names=field_names):
            kwargs = {n: None for n in _names}
            for n, v in zip(context, values, strict=False):
                kwargs[n] = v
            return _cls(**kwargs)

        with contextlib.suppress(Exception):
            pytree.register_pytree_node(cls, _flatten, _unflatten)
            _REGISTERED_DATACLASSES.add(cls)


def tree_map(f, tree, *rest):
    def _scan_and_register(obj):
        if dataclasses.is_dataclass(obj):
            _register_dataclass_as_pytree(type(obj))

            for field in dataclasses.fields(obj):
                _scan_and_register(getattr(obj, field.name))
        elif isinstance(obj, list | tuple):
            for item in obj:
                _scan_and_register(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _scan_and_register(v)

    _scan_and_register(tree)
    for t in rest:
        _scan_and_register(t)

    return pytree.tree_map(f, tree, *rest)
