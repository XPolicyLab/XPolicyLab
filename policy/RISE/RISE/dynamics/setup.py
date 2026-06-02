from setuptools import setup, find_packages
from pathlib import Path

base_dir = Path(__file__).parent
inner_dir = base_dir / "dynamics_model"

# readme_file = inner_dir / "README.md"
# long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

requirements_file = inner_dir / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="dynamics-model",
    version="0.1.0",
    description="Video prediction model for robotic dynamics",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    author="RISE Team",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "dynamics_model": [
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "*.sh",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
