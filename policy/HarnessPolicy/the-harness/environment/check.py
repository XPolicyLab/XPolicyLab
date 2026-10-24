"""Check that the harness-policy environment still has its pinned core.

Run inside the environment: python environment/check.py
It imports modules but does not start Isaac Sim or touch a GPU beyond
querying CUDA availability.
"""

from importlib import import_module, metadata
import sys

from packaging.version import Version


EXPECTED = {
    "torch": "2.7.0",
    "torchvision": "0.22.0",
    "isaacsim": "5.1.0",
    "numpy": "1.26.0",
    "websockets": "17.1",
    "transformers": "4.57.6",
}
IMPORTS = [
    "harness_policy", "isaaclab", "curobo",
    "client_server.ws.model_client", "client_server.ws.model_server",
    "cv2", "scipy", "transforms3d", "open3d", "msgpack_numpy",
    "mmengine", "liger_kernel", "accelerate", "g05", "fla",
]


def main() -> int:
    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append(f"python {sys.version.split()[0]} (expected 3.11)")
    for package, expected in EXPECTED.items():
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package} missing")
            continue
        if Version(version.split("+")[0]) != Version(expected):
            problems.append(f"{package} {version} (expected {expected})")
    for module in IMPORTS:
        try:
            import_module(module)
        except Exception as error:
            problems.append(f"import {module}: {type(error).__name__}: {error}")
    import torch
    if torch.version.cuda != "12.8":
        problems.append(f"torch CUDA {torch.version.cuda} (expected 12.8)")
    for problem in problems:
        print("FAIL", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
