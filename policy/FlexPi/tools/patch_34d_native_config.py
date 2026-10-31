"""Patch a 34-D-native run config (FastWAM or flex-pi) so the RoboDojo FlexPi adapter (yam32 bridge) can serve it.

The adapter builds a 32-wide state ([L pos3 rot6d6, R pos3 rot6d6, L grip, R grip, L joint6, R joint6]) from the
RoboDojo observation. A 34-native config (raw_shape 34, identity ScatterToChannels(total_dim=34)) cannot take it:
resolve_bridge refuses 34 vs 14. The fix: raw_shape [32, 34] in the
action/state shape_meta and AlignTo34DByWidth prepended to
action_state_transforms, in train and val. AlignTo34DByWidth zero-scatters 32 -> 34 (slots 26/33 = 0, which is what the
34-D corpus stores there) before Yam32DRelativeAction(state_dim=34), and gathers the 32 real slots back on the way out.

The AlignTo34DByWidth target is in the namespace of the chain it joins, the root package of the chain's
Yam32DRelativeAction target: fastwam.* in a FastWAM config (the vendored fastwam alias package re-exports the
flexpi class), flexpi.* in a flex-pi config. The target must import as a class from the vendored deploy code
(policy/FlexPi/FlexPi/src, first on sys.path as in the adapter); a chain in any other namespace, or a target that does
not, is refused.

Usage: python policy/FlexPi/tools/patch_34d_native_config.py <config.yaml in> <config.yaml out>
Refuses anything that is not the 34-native identity-merger layout. Pure text-free OmegaConf rewrite (no resolve).
"""
import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf

# The vendored deploy code the adapter (policy/FlexPi/model.py) puts first on sys.path.
FLEXPI_SRC = Path(__file__).resolve().parents[1] / "FlexPi" / "src"
# AlignTo34DByWidth, keyed by the root package of the chain's Yam32DRelativeAction target.
ALIGN = {
    "fastwam": "fastwam.datasets.lerobot.transforms.mixed_embodiment_align.AlignTo34DByWidth",
    "flexpi": "flexpi.datasets.lerobot.transforms.mixed_embodiment_align.AlignTo34DByWidth",
}


def align_target(yam_target, name):
    root = yam_target.split(".", 1)[0]
    if root not in ALIGN:
        raise SystemExit(f"{name}: no AlignTo34DByWidth for the {root!r} namespace of {yam_target}")
    module, _, cls = ALIGN[root].rpartition(".")
    try:
        found = getattr(importlib.import_module(module), cls, None)
    except ImportError as e:
        raise SystemExit(f"{name}: {ALIGN[root]} does not import from {FLEXPI_SRC}: {e}")
    if not isinstance(found, type):
        raise SystemExit(f"{name}: {module} in {FLEXPI_SRC} has no class {cls}")
    return ALIGN[root]


def patch_split(split, name):
    changed = []
    proc = split.processor
    merger = proc.action_state_merger
    if not str(merger._target_).endswith("ScatterToChannels") or int(merger.total_dim) != 34:
        raise SystemExit(f"{name}: merger is not ScatterToChannels(total_dim=34): {merger}")
    if merger.get("used_action_channel_ids") is not None or merger.get("used_state_channel_ids") is not None:
        raise SystemExit(f"{name}: merger carries channel ids; this is not the 34-native identity layout")
    for where in (split.shape_meta, proc.shape_meta):
        for cat in ("action", "state"):
            meta = where[cat]
            if len(meta) != 1:
                raise SystemExit(f"{name}: expected one {cat} key")
            raw = meta[0].raw_shape
            if OmegaConf.is_list(raw):
                if list(raw) != [32, 34]:
                    raise SystemExit(f"{name}: unexpected {cat} raw_shape {list(raw)}")
                continue
            if int(raw) != 34:
                raise SystemExit(f"{name}: {cat} raw_shape {raw} is not 34")
            meta[0].raw_shape = [32, 34]
            changed.append(f"{cat}.raw_shape 34 -> [32, 34]")
    chain = proc.action_state_transforms
    targets = [str(t._target_) for t in chain]
    if not targets or not targets[-1].endswith("Yam32DRelativeAction") or int(chain[-1].get("state_dim", 32)) != 34:
        raise SystemExit(f"{name}: chain {targets} does not end in Yam32DRelativeAction(state_dim=34)")
    if not any(t.endswith("AlignTo34DByWidth") for t in targets):
        align = align_target(targets[-1], name)
        proc.action_state_transforms = [OmegaConf.create({"_target_": align})] + list(chain)
        changed.append("AlignTo34DByWidth prepended to action_state_transforms")
    return changed


sys.path.insert(0, str(FLEXPI_SRC))
cfg = OmegaConf.load(sys.argv[1])
report = {}
for split in ("train", "val"):
    if split in cfg.data and cfg.data[split] is not None:
        report[split] = patch_split(cfg.data[split], f"data.{split}")
OmegaConf.save(cfg, sys.argv[2])
for k, v in report.items():
    print(f"data.{k}: " + ("; ".join(v) if v else "already patched"))
