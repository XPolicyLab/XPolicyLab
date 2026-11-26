"""Prepare pinned PRO metadata beside an isolated rpent-liberopro installation."""

from pathlib import Path
import shutil
from .protocols import REVISIONS


def prepare(runtime, assets, snapshot=None):
    root = Path(runtime).resolve()
    core = root / "liberopro/liberopro"
    if not (core / "__init__.py").is_file():
        raise FileNotFoundError("Install rpent-liberopro==0.2.0 into --runtime first")
    asset_root = Path(assets).resolve()
    for name in (
        "libero_tabletop_base_style.xml",
        "libero_floor_base_style.xml",
        "libero_kitchen_tabletop_base_style.xml",
        "libero_living_room_tabletop_base_style.xml",
    ):
        if not (asset_root / "scenes" / name).is_file():
            raise FileNotFoundError(asset_root / "scenes" / name)
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            "zhouxueyang/LIBERO-Pro", repo_type="dataset", revision=REVISIONS["pro"]
        )
    snapshot = Path(snapshot)
    # The released dataset may wrap these directories in libero_data/.
    for directory, marker in [
        ("init_files", "AUTHORITATIVE_INIT_REVISION"),
        ("bddl_files", "AUTHORITATIVE_BDDL_REVISION"),
    ]:
        sources = [p for p in snapshot.rglob(directory) if p.is_dir()]
        if len(sources) != 1:
            raise ValueError(f"Expected exactly one {directory} tree in {snapshot}")
        # Copy authoritative metadata, including the three repaired Task BDDLs.
        shutil.copytree(sources[0], core / directory, dirs_exist_ok=True)
        (root / marker).write_text(REVISIONS["pro"] + "\n")
    destination = core / "assets"
    if destination.is_symlink() and destination.resolve() != asset_root:
        raise FileExistsError(f"Existing assets symlink points elsewhere: {destination}")
    if not destination.exists():
        destination.symlink_to(asset_root, target_is_directory=True)
    elif destination.resolve() != asset_root:
        shutil.copytree(asset_root, destination, dirs_exist_ok=True)
    return dict(
        runtime=str(root),
        revision=REVISIONS["pro"],
        assets=str(destination),
        note="All 80 environment reset checks run before policy rollout in the evaluation workers",
    )
