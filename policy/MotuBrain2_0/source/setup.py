from pathlib import Path

from setuptools import find_namespace_packages, setup

root = Path(__file__).parent
package_dir = {"motubrain_robodojo": "motubrain_robodojo"}
for source in ("ss-soma", "ss-falcon", "ss-morph", "ss-modelzoo"):
    for package in find_namespace_packages(source, exclude=["*.__pycache__"]):
        package_dir[package] = str(Path(source, *package.split(".")))

setup(
    packages=list(package_dir),
    package_dir=package_dir,
    package_data={"motubrain_robodojo": ["assets/*.yaml"]},
    install_requires=[
        line.split("#", 1)[0].strip()
        for line in (root / "requirements-policy.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ],
)
