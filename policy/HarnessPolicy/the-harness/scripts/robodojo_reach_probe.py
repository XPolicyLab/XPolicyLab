#!/usr/bin/env python3
"""Map where each arm has IK solutions, for two approach orientations.

    python scripts/robodojo_reach_probe.py --endpoint http://127.0.0.1:8770

Resets one episode (does not act), then asks the worker for IK on a grid of
link6 positions over the table. Prints a character map per height:
'#' both orientations, 'f' forward only, 'd' down only, '.' none.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.core.robot import RemoteEnvironment


START = [0.707, 0.0, 0.0, 0.707]  # wxyz: link6 +x points along world +y


def down_quaternion():
    """Start orientation turned so link6 +x (the approach axis) points along world -z."""
    start = Rotation.from_quat([START[1], START[2], START[3], START[0]])
    turned = Rotation.from_euler("x", -90, degrees=True) * start
    x, y, z, w = turned.as_quat()
    assert np.allclose(turned.as_matrix()[:, 0], [0, 0, -1], atol=1e-3)
    return [float(w), float(x), float(y), float(z)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8770")
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = RemoteEnvironment(args.endpoint, timeout=600)
    env.tools()
    episode = env.reset(args.task, args.seed).episode_id
    down = down_quaternion()
    print("down quaternion (wxyz):", [round(v, 4) for v in down])
    xs = np.round(np.arange(-0.6, 0.61, 0.1), 2)
    ys = np.round(np.arange(-0.4, 0.31, 0.1), 2)
    for arm in ("left_arm", "right_arm"):
        for z in (0.80, 0.90, 1.00):
            print(f"\n{arm} z={z}  (rows y from {ys[-1]} down to {ys[0]}, columns x {xs[0]}..{xs[-1]})")
            for y in ys[::-1]:
                row = ""
                for x in xs:
                    forward = env.call("ik", episode_id=episode, arm=arm, pose=[x, y, z] + START)["success"]
                    downward = env.call("ik", episode_id=episode, arm=arm, pose=[x, y, z] + down)["success"]
                    row += "#" if forward and downward else "f" if forward else "d" if downward else "."
                print(f"  y={y:+.1f} {row}")


if __name__ == "__main__":
    main()
