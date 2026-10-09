# Copyright (C) 2026 Xiaomi Corporation.
from typing import List

from setuptools import find_packages, setup


def fetch_requirements(paths) -> List[str]:
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
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


setup(
    name="Xiaomi-Robotics-1",
    version="1.0.0",
    description="Post-training and deployment for Xiaomi-Robotics-1",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://robotics.xiaomi.com/xiaomi-robotics-1.html",
    project_urls={
        "Source": "https://github.com/XiaomiRobotics/Xiaomi-Robotics-1",
    },
    packages=find_packages(exclude=["configs", "configs.*"]),
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
