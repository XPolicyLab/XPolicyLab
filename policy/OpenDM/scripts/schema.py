"""Source-export schema checks shared by the converter and its tests (no heavy deps)."""

from XPolicyLab.utils import wuji58

JOINT14_NAMES = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]
SUPPORTED_CONTRACTS = {
    ("RoboDojo", "arx_x5", "joint"): "joint14",
    ("RoboDojo", "tianji_wuji", "ee"): "wuji58",
}


def contract_name(bench_name, env_cfg_type, action_type):
    try:
        return SUPPORTED_CONTRACTS[(bench_name, env_cfg_type, action_type)]
    except KeyError:
        supported = ", ".join("/".join(key) for key in SUPPORTED_CONTRACTS)
        raise ValueError(
            f"Unsupported contract {bench_name}/{env_cfg_type}/{action_type}; supported: {supported}"
        ) from None


def validate_source_info(info, contract, action_dim):
    """Check a LeRobot ``meta/info.json`` against the selected contract; return nothing."""
    if info["fps"] != 25:
        raise ValueError("The released history recipe requires 25 Hz data")
    if contract == "joint14":
        names = JOINT14_NAMES
    elif contract == "wuji58":
        names = list(wuji58.MOTOR_NAMES)
        repr_mode = info.get("repr_mode")
        if repr_mode != wuji58.REPR_MODE:
            raise ValueError(
                f"wuji58 needs an export converted with --representation wuji58 "
                f"(info.json repr_mode={wuji58.REPR_MODE!r}), got {repr_mode!r}"
            )
        if info.get("robot") not in (None, "tianji_wuji"):
            raise ValueError(f"wuji58 export was recorded on {info.get('robot')!r}, not tianji_wuji")
    else:
        raise ValueError(f"unknown contract {contract!r}")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature["names"] != [names]:
            raise ValueError(f"{key} names do not match the {contract} contract; use the matching export")
        if list(feature["shape"]) != [action_dim]:
            raise ValueError(f"{key} shape {feature['shape']} != [{action_dim}]")
