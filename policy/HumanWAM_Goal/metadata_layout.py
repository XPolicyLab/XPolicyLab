"""Validate the single metadata root used by XPolicyLab's official lookup."""

from pathlib import Path


def validate_metadata_layout(xpolicylab_root, robotwin_root):
    xpl = Path(xpolicylab_root).resolve()
    declared = Path(robotwin_root).resolve() / "env_cfg"
    # Official process_data.py derives ../../env_cfg from its own __file__.
    actual = (xpl / "utils/process_data.py").parent / "../../env_cfg"
    if not declared.is_dir() or not actual.is_dir() or actual.resolve() != declared.resolve():
        raise ValueError(
            "XPolicyLab's parent/env_cfg must resolve to ROBOTWIN_ROOT/env_cfg. "
            "Use a checkout under that RoboTwin root, or an existing explicit same-source metadata symlink; "
            "the installer does not copy or replace env_cfg."
        )
    for relative in ("arx_x5.yml", "aloha_agilex.yml", "robot/_robot_info.json"):
        path = actual / relative
        if not path.is_file() or path.resolve() != (declared / relative).resolve():
            raise ValueError("Missing or inconsistent official robot metadata: " + relative)
    return str(actual.resolve())


def validate_runtime_lookup(xpolicylab_root, robotwin_root):
    root = validate_metadata_layout(xpolicylab_root, robotwin_root)
    from XPolicyLab.utils import process_data

    expected = Path(xpolicylab_root).resolve() / "utils/process_data.py"
    if Path(process_data.__file__).resolve() != expected.resolve():
        raise ValueError("Imported XPolicyLab lookup is from a different checkout")
    aloha = process_data.get_robot_action_dim_info("aloha_agilex")
    if process_data.get_robot_action_dim_info("arx_x5") != aloha:
        raise ValueError("Official arx_x5 and aloha_agilex wire dimensions differ")
    return dict(env_cfg_root=root, dimensions=aloha, lookup_file=str(expected))
