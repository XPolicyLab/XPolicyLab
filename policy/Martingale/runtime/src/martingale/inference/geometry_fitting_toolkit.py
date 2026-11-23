from pathlib import Path

import numpy as np

from .geometry_fitting import GeometryFitting
from .geometry_fitting_specs import examples
from .isolation import CellWorker
from .motion_preview import preview
from .toolkit import Toolkit, _render_extension


class FittingToolkit(Toolkit):
    METHODS = frozenset({"fit_requirements", "fit_require", "fit_progress", "fit_image", "fit_triangulate",
                         "fit_estimate", "fit_unresolved", "fit_impossible", "fit_finish"})
    READS = frozenset({"display", "action_count", "task", "record_progress", "state"})

    def __init__(self, robo, workspace, *, task, worker=None):
        Path(workspace).mkdir(parents=True, exist_ok=True)
        worker = worker or CellWorker(workspace, include_library=False)
        super().__init__(robo, workspace, worker=worker)
        self.fitting = GeometryFitting(task, workspace)
        self.calibration_complete = False

    def fit_capture(self, cameras=("cam_head", "cam_left_wrist", "cam_right_wrist")):
        """Save current original RGB and camera calibration, returning view IDs and metadata.

        No actions. Use display(robo.fit_image(view_id)) to inspect saved pixels.
        All cameras in a capture share one recorded observation step. Corresponding
        object landmarks must remain static between captures.
        """
        return self.fitting.capture(self.robo, cameras)

    def fit_segment(self, view_id, text_prompt):
        """Segment one SAVED RGB view using SAM3, returning mask and confidence.

        No geometry priors, depth or task-specific appearance template. Check the
        mask against RGB for identity and occlusion before using its contour.
        """
        if not isinstance(text_prompt, str) or not 1 <= len(text_prompt.strip()) <= 256:
            raise ValueError("Provide a concise object description")
        rgb = self.fitting.fit_image(view_id)
        sam = self.robo._backend._toolkit._primitives.sam3_client
        result = sam.segment(rgb, text_prompt=text_prompt, min_score=0.2)
        return {"view_id": view_id, "text_prompt": text_prompt, "found": bool(result.found),
                "mask": np.asarray(result.mask, bool) if result.found else None,
                "score": float(result.score) if result.score is not None else None}

    def move_to(self, xyz, arm="right", pitch=None, yaw=None, roll=None):
        """Move an EMPTY OPEN hand to obtain an RGB view, along a straight path.

        xyz is the grasp point in world metres, arm is left/right. pitch/yaw/roll
        are degrees (None keeps the angle): pitch=0 points horizontally, 90 down;
        yaw=0 points away from the robot, +90 left, -90 right. Inspect RGB and
        preview_move first. This is for observation only; no object contact.
        """
        self.fitting._mutable()
        return self.robo.move_to(xyz, arm=arm, gripper=-1, pitch=pitch, yaw=yaw, roll=roll, path="line")

    def tool_doc(self, *names):
        functions = {name: getattr(self.fitting, name) for name in self.METHODS}
        functions.update(fit_capture=self.fit_capture, fit_segment=self.fit_segment,
                         move_to=self.move_to, preview_move=preview, state=self.robo.state)
        if not names:
            return "Calibration tools: " + ", ".join(sorted({*functions, *self.READS}))
        result = []
        for name in names:
            if name not in functions:
                raise PermissionError("Unavailable calibration tool: " + str(name))
            result.append(_render_extension(name, functions[name]))
        return "\n\n".join(result)

    def prompt(self, instruction):
        return "\n\n".join([
            Path(__file__).with_name("GEOMETRY_FITTING.md").read_text(),
            "World frame: x across table (left arm at negative x), y away from robot, z up. "
            "Units metres; quaternions [w,x,y,z]. Cameras look along local -z with local +y up. "
            "state() returns robot arms only (tcp_pos, eef_pos, eef_quat, joint angles). "
            "do not waste actions",
            self.tool_doc(*sorted(self.METHODS), "fit_capture", "fit_segment", "move_to", "preview_move", "state"),
            "Measurement examples for ALL tasks (no reference values):\n" + examples(),
            "Current calibration task: " + self.fitting.task,
            instruction,
        ])

    def dispatch(self, name, args, kwargs):
        # Enforce the calibration API allowlist at RPC dispatch.
        if name in self.METHODS:
            return getattr(self.fitting, name)(*args, **kwargs)
        if name in {"fit_capture", "fit_segment", "tool_doc", "move_to"}:
            return getattr(self, name)(*args, **kwargs)
        if name == "preview_move":
            return preview(self.robo, *args, **kwargs)
        if name in self.READS:
            return super().dispatch(name, args, kwargs)
        raise PermissionError("Tool unavailable during blind RGB fitting: " + name)

    def execute_tool(self, name, arguments):
        result = super().execute_tool(name, arguments)
        # Signal completion after the final cell receipt has been persisted.
        self.calibration_complete = self.fitting.finished and not self.suspended
        return result
