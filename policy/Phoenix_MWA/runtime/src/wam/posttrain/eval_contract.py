"""Execution-time inverse of the FLU/B=I LIBERO training transform."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from wam.datasets.lerobot.transforms.libero_flu_bi import LIBERO_FLU_BI_CONTRACT, LiberoRight80FLUBIdentityTransform


def validate_evaluation_contract(cfg, payload, stats_path, *, expected_pretrain_step=None):
    """Do not evaluate old AB weights using the new inverse transform."""
    version = LIBERO_FLU_BI_CONTRACT["version"]
    if cfg.joint96_posttrain.libero_frame_contract != version:
        raise ValueError("Expected the new FLU/B=I posttraining configuration")
    for split in ("train", "val"):
        target = cfg.data[split].processor.action_state_transforms[0]._target_
        if target != "wam.datasets.lerobot.transforms.libero_flu_bi.LiberoRight80FLUBIdentityTransform":
            raise ValueError("Wrong proprio/action coordinate transform")
    raw = Path(stats_path).read_bytes()
    def float32_document(value):
        # The existing training-state writer reloads JSON numbers as float32
        # tensors before exporting them. Require equality at that precision,
        # preserving dictionary keys, strings and list ordering exactly.
        if isinstance(value, dict):
            return {k: float32_document(v) for k, v in value.items()}
        if isinstance(value, list):
            return [float32_document(v) for v in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(value):
                raise ValueError("Nonfinite statistics metadata")
            return float(np.float32(value))
        return value
    document = json.loads(raw)
    if float32_document(document["frame_contract"]) != float32_document(json.loads(json.dumps(LIBERO_FLU_BI_CONTRACT))):
        raise ValueError("Wrong normalization frame convention")
    contract = payload.get("joint96_metadata", {}).get("soft_prompt", {}).get("training_contract", {})
    if (contract.get("frame_contract") != version
            or contract.get("embodiment_id") != 0 or contract.get("domain") != "libero"):
        raise ValueError("Checkpoint frame/prompt/stat provenance mismatch")


def decode_at_execution(canonical, obs):
    from eval.libero.eval_libero_single import quat2axisangle
    value = np.asarray(canonical, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite canonical 7D action")
    # LIBERO obs quaternion is XYZW, exactly as the existing proprio adapter.
    ee = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"])))
    if ee.shape != (6,) or not np.isfinite(ee).all():
        raise ValueError("Invalid measured source EE pose")
    return LiberoRight80FLUBIdentityTransform().decode_simulator_commands(
        torch.as_tensor(value).reshape(1, 7), torch.as_tensor(ee, dtype=torch.float32).reshape(1, 6)
    ).numpy()[0].tolist()
