"""Install application dependencies while preserving the selected framework."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Select a Python 3.10–3.12 inference environment.")
    selected = {name: metadata.version(name) for name in ("torch", "torchvision")}
    destination = ROOT / "source_inference" / ".venv"
    if not destination.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(destination)],
            check=True,
        )
    python = destination / "bin" / "python"
    requirements = [ROOT / "requirements.txt"]
    runtime_requirements = ROOT / "source_inference" / "requirements.txt"
    if runtime_requirements.is_file():
        requirements.append(runtime_requirements)
    with tempfile.TemporaryDirectory(prefix="cloudrobo-install-") as temporary:
        constraints = Path(temporary) / "constraints.txt"
        constraints.write_text("".join(f"{name}==={version}\n" for name, version in selected.items()))
        command = [str(python), "-m", "pip", "install", "--constraint", str(constraints)]
        # Compatibility flag: pip cache/network policy is configured by the user.
        for requirement in requirements:
            command.extend(["--requirement", str(requirement)])
        subprocess.run(command, check=True)
    probe = (
        "import json; from importlib import metadata; "
        "print(json.dumps({n:metadata.version(n) for n in ('torch','torchvision')}))"
    )
    actual = json.loads(subprocess.check_output([str(python), "-c", probe], text=True))
    if actual != selected:
        raise RuntimeError("The selected framework changed during installation.")
    print(f"Inference interpreter: {python}")


if __name__ == "__main__":
    main()
