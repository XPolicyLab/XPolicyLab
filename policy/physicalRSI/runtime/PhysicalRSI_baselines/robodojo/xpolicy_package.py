"""Install a portable adapter with skill implementations and asset download metadata."""

import argparse
import shutil
from importlib.resources import files
from pathlib import Path


def install(checkout, *, native_client=False):
    checkout = Path(checkout).resolve()
    if native_client:
        if not (checkout / "src/eval_client/main.py").is_file():
            raise ValueError("Expected a RoboDojo simulator workspace")
        checkout = checkout / "XPolicyLab"
    elif not (checkout / "model_template.py").is_file():
        raise ValueError("Expected an XPolicyLab checkout")
    destination = checkout / "policy/physicalRSI"
    source = files("PhysicalRSI_baselines.robodojo.xpolicylab_policy")
    entries = [
        entry
        for entry in source.iterdir()
        if entry.is_file()
        and entry.name != "__pycache__"
        and (
            not native_client
            or entry.name in {"__init__.py", "deploy.py", "deploy.yml"}
        )
    ]
    if destination.exists():
        for entry in entries:
            target = destination / entry.name
            if target.exists() and target.read_bytes() != entry.read_bytes():
                raise FileExistsError(
                    f"Refusing to replace changed adapter file: {target}"
                )
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        with (
            entry.open("rb") as incoming,
            (destination / entry.name).open("wb") as outgoing,
        ):
            shutil.copyfileobj(incoming, outgoing)
        shutil.copymode(entry, destination / entry.name)
    if not native_client:
        # Carry the repository runtime with the adapter. No editable source checkout
        # or developer-specific PYTHONPATH is required on the evaluator machine.
        repository = Path(__file__).resolve().parents[2]
        for package in ("PhysicalRSI", "PhysicalRSI_core", "PhysicalRSI_baselines"):
            source_root = repository / package
            for incoming in source_root.rglob("*"):
                relative = incoming.relative_to(source_root)
                if any(part in {"releases", "__pycache__", ".pytest_cache"} for part in relative.parts):
                    continue
                if not incoming.is_file() or incoming.suffix not in {".py", ".json", ".yml", ".yaml"} and not incoming.name.startswith(("LICENSE", "NOTICE")):
                    continue
                target = destination / "runtime" / package / relative
                if target.exists() and target.read_bytes() != incoming.read_bytes():
                    raise FileExistsError(f"Refusing to replace changed runtime: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(incoming, target)
        release = Path(__file__).parent / "releases/robodojo-v1"
        for name in ("download_assets.py", "assets.json"):
            incoming = release / name
            # Portable source exports carry these files beside the adapter.
            if not incoming.is_file():
                incoming = Path(str(source)) / name
            if incoming.is_file():
                target = destination / name
                if target.exists() and target.read_bytes() != incoming.read_bytes():
                    raise FileExistsError(f"Refusing to replace changed asset metadata: {target}")
                shutil.copyfile(incoming, target)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    parser.add_argument("--native-client", action="store_true")
    args = parser.parse_args()
    print(install(args.checkout, native_client=args.native_client))
