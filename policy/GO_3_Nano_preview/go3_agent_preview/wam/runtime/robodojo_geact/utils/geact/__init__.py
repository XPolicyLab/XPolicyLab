import os
import sys
import importlib
import importlib.util
from typing import Union, Type
import torchvision
from einops import rearrange
import torch


def import_custom_class(
    class_name: str,
    source: Union[str, os.PathLike],
) -> Type:
    if not os.path.isabs(source) and not source.endswith('.py'):
        try:
            module = importlib.import_module(source)
            if hasattr(module, class_name):
                return getattr(module, class_name)
            else:
                raise ImportError(
                    f"Class '{class_name}' not found in module '{source}'. "
                    f"Available symbols: {[attr for attr in dir(module) if not attr.startswith('__')]}"
                )
        except ImportError:
            if "/" not in str(source) and "\\" not in str(source):
                raise

    file_path = os.path.abspath(source)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    unique_name = module_name

    if unique_name in sys.modules:
        module = sys.modules[unique_name]
    else:
        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None:
            raise ImportError(f"Failed to create module spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[unique_name]
            raise RuntimeError(f"Module execution failed: {e}") from e

    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        available = [attr for attr in dir(module) if not attr.startswith('__')]
        raise AttributeError(
            f"Class '{class_name}' not found in file '{file_path}'. "
            f"Available symbols: {', '.join(available)}"
        )


def save_video(tensor, save_path, fps=30):
    tensor = ((tensor + 1) / 2 * 255).to(torch.uint8)
    tensor = rearrange(tensor, "c t h w -> t h w c")
    torchvision.io.write_video(save_path, tensor, fps=fps)
