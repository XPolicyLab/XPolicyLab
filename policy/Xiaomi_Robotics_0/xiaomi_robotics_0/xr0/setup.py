# Copyright (C) 2026 Xiaomi Corporation.
from typing import List

from setuptools import find_packages, setup


def fetch_requirements(paths) -> List[str]:
    """
    Reads one or more requirements files and returns a list of requirements.

    Args:
        paths (Union[str, List[str]]): Path(s) to requirements file(s).

    Returns:
        List[str]: List of requirements.
    """
    if not isinstance(paths, list):
        paths = [paths]
    requirements = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fd:
                requirements += [r.strip() for r in fd if r.strip() and not r.startswith("#")]
        except FileNotFoundError:
            pass
    return requirements


def fetch_readme() -> str:
    """
    Reads the README.md file in the current directory and returns its content.
    """
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


setup(
    name="MiBot",
    version="1.0.0",
    description="Democratizing Highly Efficient Robot Training",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="",
    project_urls={
        "Github": "",
    },
    packages=find_packages(
        exclude=[
            "assets",
            "cache",
            "configs",
            "docs",
            "eval",
            "evaluation_results",
            "gradio",
            "logs",
            "notebooks",
            "outputs",
            "pretrained_models",
            "samples",
            "scripts",
            "tests",
            "tools",
            "tmp",
            "*.egg-info",
        ]
    ),
    install_requires=fetch_requirements("assets/requirements.txt"),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
