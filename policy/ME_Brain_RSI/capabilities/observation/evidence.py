"""Record only the observation and emitted actions at the policy boundary."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def snapshot(obs, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, camera in obs.get("vision", {}).items():
        color = np.asarray(camera["color"], dtype=np.uint8)
        Image.fromarray(color[..., :3]).save(folder / (name + ".png"))
    allowed = {k: obs[k] for k in ("state", "instruction", "instructions") if k in obs}
    (folder / "observation.json").write_text(json.dumps(plain(allowed), indent=2))
