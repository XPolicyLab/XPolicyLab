"""Per-execution replay artifacts, laid out like H4/H5 observations so the L20 console replays them alike.

run_dir/observations/000000/            the scene before the first program
run_dir/observations/<n>/execution_cam_<camera>.mp4   frames recorded while program n ran
run_dir/observations/<n>/current_cam_<camera>.png     the view after it
run_dir/observations/<n>/state.json                   arm state after it (RoboDojo state keys)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ExecutionArtifacts:
    def __init__(self, runtime, frames, out_dir: Path, fps: float):
        self.rt = runtime
        self.frames = frames
        self.out = Path(out_dir)
        self.fps = float(fps)
        self.start: dict[str, int] = {}

    def state(self) -> dict:
        out = {}
        for arm, st in self.rt.obs.arms.items():
            opening = st.gripper_opening if st.gripper_opening is not None else st.gripper_command
            out[f"{arm}_ee_pose"] = [round(float(v), 5) for v in st.ee_pose.array()]
            out[f"{arm}_arm_joint_state"] = [round(float(v), 5) for v in np.asarray(st.joints).reshape(-1)] \
                if st.joints is not None else None
            out[f"{arm}_ee_joint_state"] = [round(float(opening), 4)] if opening is not None else None
        return out

    def _snapshot(self, folder: Path) -> None:
        from PIL import Image

        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in self.rt.obs.cameras.items():
            Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(folder / f"current_cam_{name}.png")
        (folder / "state.json").write_text(json.dumps({"tick": self.rt.tick, "state": self.state()}), encoding="utf-8")

    def begin(self) -> None:
        if not (self.out / "000000").exists():
            self._snapshot(self.out / "000000")
        self.start = {cam: len(images) for cam, images in self.frames.frames.items()}

    def end(self, n: int) -> None:
        from knowin_agent.backends.robodojo.video import write_videos

        folder = self.out / f"{int(n):06d}"
        clips = {cam: images[self.start.get(cam, 0):] for cam, images in self.frames.frames.items()}
        write_videos({cam: c for cam, c in clips.items() if c}, folder, fps=self.fps, prefix="execution_cam_")
        self._snapshot(folder)
