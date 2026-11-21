"""RoboDojo adapter with a public step budget and optional private replay prelude."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapter import Adapter as BaseAdapter


class Adapter(BaseAdapter):
    def __init__(self, environment_dir, options):
        self.replay = options.get("replay")
        self.replay_consumed = False
        super().__init__(environment_dir, options)

    def _transition(self, applied_action=None):
        result = super()._transition(applied_action)
        observation = result["observation"]
        observation["step_limit"] = int(self.env.step_lim)
        observation["steps_remaining"] = max(0, observation["step_limit"] - observation["control_step"])
        return result

    def reset(self, episode, record_dir, observation_dir):
        result = super().reset(episode, record_dir, observation_dir)
        if self.replay is None or self.replay_consumed or episode["id"] != self.replay["episode_id"]:
            return result

        source = Path(self.replay["source_record"]).resolve()
        source_episode = json.loads((source / "episode.json").read_text())
        for key in ("task", "robot", "seed", "layout_id"):
            if source_episode[key] != episode[key]:
                raise ValueError(f"Replay episode differs in {key}")
        target_step = int(self.replay["control_step"])
        if not 0 < target_step < self.env.step_lim:
            raise ValueError("Replay checkpoint must precede native termination")

        action_records = [(source, target_step)]
        while True:
            record = action_records[-1][0]
            parent_config = json.loads((record.parent.parent / "config.json").read_text())
            parent_replay = parent_config["options"].get("replay")
            if parent_replay is None or parent_replay["episode_id"] != json.loads((record / "episode.json").read_text())["id"]:
                break
            parent = Path(parent_replay["source_record"]).resolve()
            if any(record == parent for record, _ in action_records):
                raise ValueError("Replay action history contains a cycle")
            action_records.append((parent, int(parent_replay["control_step"])))
        action_records.reverse()
        action_files = [record / "actions.jsonl" for record, _ in action_records]
        started = time.monotonic()
        count = 0
        for (_, cutoff), action_file in zip(action_records, action_files):
            # A resumed controller may stop before issuing any new action.
            # Its parent must already supply the complete checkpoint.
            if not action_file.exists():
                if count != cutoff:
                    raise ValueError("Missing replay action file before checkpoint")
                continue
            for row in map(json.loads, action_file.read_text().splitlines()):
                if row.get("status") != "completed":
                    continue
                if row["control_step"] > cutoff:
                    break
                if row["control_step"] != count + 1:
                    raise ValueError("Replay action sequence is incomplete")
                result = self._step_joint(row["applied_action"]["joint_command"])
                count += 1
                if result["observation"]["control_step"] != count or result["done"]:
                    raise RuntimeError(f"Replay diverged or ended at control step {count}")
        if count != target_step:
            raise ValueError(f"Replay contains {count} of {target_step} requested steps")

        expected = None
        for row in map(json.loads, (source / "transitions.jsonl").open()):
            if row["observation"]["control_step"] == target_step:
                expected = row["observation"]
                break
        if expected is None:
            raise ValueError("Original checkpoint observation is missing")
        current = result["observation"]
        pose_error_m = max(
            float(np.linalg.norm(np.asarray(current["state"][f"{arm}_ee_pose"][:3])
                                 - np.asarray(expected["state"][f"{arm}_ee_pose"][:3])))
            for arm in ("left", "right")
        )
        angle_error_rad = max(
            float(2 * np.arccos(np.clip(abs(np.dot(
                np.asarray(current["state"][f"{arm}_ee_pose"][3:]),
                np.asarray(expected["state"][f"{arm}_ee_pose"][3:])
            )), 0, 1)))
            for arm in ("left", "right")
        )
        original_image = source.parent.parent / "observations" / source.name / f"{target_step:05d}_cam_head.png"
        replay_image = Path(observation_dir) / f"{target_step:05d}_cam_head.png"
        a = np.asarray(Image.open(original_image), dtype=np.int16)
        b = np.asarray(Image.open(replay_image), dtype=np.int16)
        rgb_mean_error = float(np.abs(a - b).mean()) if a.shape == b.shape else float("inf")
        rgb_tolerance = float(self.replay.get("head_rgb_tolerance", 6.0))
        validation = {
            "source_record": str(source),
            "source_actions_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in action_files if path.exists())).hexdigest(),
            "replayed_steps": count,
            "replay_wall_seconds": time.monotonic() - started,
            "eef_position_max_error_m": pose_error_m,
            "eef_orientation_max_error_rad": angle_error_rad,
            "head_rgb_mean_absolute_error": rgb_mean_error,
            "head_rgb_tolerance": rgb_tolerance,
            "matched": pose_error_m <= 0.015 and angle_error_rad <= 0.15 and rgb_mean_error <= rgb_tolerance,
        }
        (Path(record_dir) / "replay_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        if not validation["matched"]:
            raise RuntimeError("Replayed checkpoint does not match saved RGB/proprioception")
        self.replay_consumed = True
        return result
