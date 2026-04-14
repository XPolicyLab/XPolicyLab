import importlib.metadata
import importlib.util
import os
import re
from typing import List

from setuptools import find_packages, setup


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_torch_npu_available() -> bool:
    return _is_package_available("torch_npu")


def _is_torch_available() -> bool:
    return _is_package_available("torch")


def _is_torch_cuda_available() -> bool:
    if _is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def get_version() -> str:
    with open(os.path.join("lingbotvla", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines

BASE_REQUIRE = [
    "torchdata>=0.8.0,<1.0",
    "blobfile>=3.0.0",
]

def main():
    # Update install_requires and extras_require
    install_requires = BASE_REQUIRE

    setup(
        name="lingbotvla",
        version=get_version(),
        python_requires=">=3.8.0",
        packages=find_packages(exclude=["scripts", "tasks", "tests"]),
        url="https://www.robbyant.com",
        license="Apache 2.0",
        author="Robbyant Team",
        author_email="lf419501@antgroup.com",
        description="LingBot-VLA: A Pragmatic VLA Foundation Model",
        install_requires=install_requires,
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
