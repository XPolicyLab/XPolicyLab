from functools import lru_cache
import hashlib
import json
from pathlib import Path

from skills.shared.motion.urdf_pose_kinematics.implementation import OwnURDFKinematics


@lru_cache(maxsize=1)
def run():
    root = Path(__file__).resolve().parents[4]
    settings = json.loads((root / 'configs/mature-trajectory.json').read_text())
    source = root / settings['robot_urdf']
    if hashlib.sha256(source.read_bytes()).hexdigest() != settings['robot_urdf_sha256']:
        raise ValueError('The declared own-robot numerical model changed')
    return OwnURDFKinematics(source)
