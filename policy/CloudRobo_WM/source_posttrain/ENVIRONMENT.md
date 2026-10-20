# Reproducible environment

The Python package environment uses Python 3.12 on Linux. The required framework base versions are PyTorch 2.9.1 and torchvision 0.24.1. Install the framework builds and any device extension from the distribution supported by your selected runtime before installing this project. The project does not select a device runtime for you.

`runtime-requirements.toml` records the framework base versions and the application packages Accelerate 1.12.0 and timm 1.0.26. The remaining Python dependencies, including tqdm, are declared in `pyproject.toml`. Use `python tools/install_dependencies.py` after preparing the framework and decoder. It checks the installed framework base versions and constrains dependency resolution to their existing complete version strings. It stops if the framework or decoder is missing rather than choosing another framework build. Direct installation of the project alone does not install the separately managed framework or application packages.

RGB and packed-depth frame selection require a compatible CPU build exposing the `decord` module, such as [decord2](https://github.com/johnnynunez/decord2). The current bundle does not include a decoder binary or claim that a stock package reproduces the original custom build. Supply the decoder separately and validate frame selection, timestamps and lossless packed-depth reconstruction with `crwm.precheck` on the target system. The prerequisite check only establishes that the module is available; it does not validate its behavior.

Only cached text tensors are needed for training; a T5 tokenizer/text encoder, online model downloads and evaluation simulators are not training dependencies.

Device visibility and runtime libraries are configured by the launch environment. Configure any backend-required determinism settings before starting training. Strict deterministic algorithms remain enabled; unsupported deterministic operations raise an error. Device compatibility depends on the installed framework and runtime and must be validated on the target system.
