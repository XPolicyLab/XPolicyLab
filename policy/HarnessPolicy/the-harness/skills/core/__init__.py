"""State, camera views, show() and finish(): what every program needs."""

import base64

from harness_policy.runtime.events import Escalation, ProgramFinished
from harness_policy.runtime.session import Observation


class Core:
    def finish(self, summary=""):
        """End the episode now: the program's own checks say the task is done. No
        further model call follows."""
        raise ProgramFinished(summary)

    def _wrap_up(self):
        """After the episode is finished (finish() or <finish>), for benchmarks that
        count success only with the arms back at their start pose (profile
        return_to_start, RoboDojo): open each gripper that holds nothing, then move both
        arms back to their start joints together (home). RoboDojo scores 23 of its 42
        tasks only with every gripper at least 0.8 open (is_all_gripper_open), so jaws
        left closed on nothing (after a push, say) would lose the whole score; a gripper
        that holds something stays closed, since the lifting tasks (general_pickup,
        match_and_pick_from_conveyor) score the object still in the air (Z2 left every
        gripper as it was). Returns the end event of each step, HOLDING for a gripper
        kept closed."""
        if not self.profile.get("return_to_start"):
            return []
        self.__dict__["_wrapping_up"] = True  # the benchmark's convention, not a motion of the program
        kinds = []
        for arm in ("left_arm", "right_arm"):
            if self._holds_at_end(arm):
                kinds.append("HOLDING")
                continue
            try:
                kinds.append(self.open_gripper(arm).event.kind)
            except Escalation as escalation:
                kinds.append(escalation.trace.event.kind)
        try:
            traces = self.home(("left_arm", "right_arm"))
        except Escalation as escalation:
            traces = getattr(escalation, "traces", None) or [escalation.trace] * 2
        return kinds + [trace.event.kind for trace in traces]

    def _holds_at_end(self, arm):
        """Whether the wrap-up keeps the gripper of `arm` closed: it was last commanded
        closed and holding() says it holds something (the measured opening, or the vision
        model's look at the wrist camera); when holding() cannot tell, a grip the pick
        recorded, else closed as it is."""
        if getattr(self, "_gripper_commands", {}).get(arm, 1.0) > 0.5:
            return False
        try:
            return bool(self.holding(arm))
        except Exception:
            grips = self.__dict__.get("_eef_grip")
            return arm in grips if grips is not None else True

    def get_state(self):
        """The robot state, read fresh from the robot: step, ended, joint_position (14: left
        6 joints, left gripper, right 6 joints, right gripper), left_ee_pose and
        right_ee_pose (link6 poses), and step_limit when the connection reports it."""
        self.state = self.env.call("state", episode_id=self.episode_id)
        return self.state

    def get_observation(self, cameras=None):
        """Current camera views as an Observation: .images[camera] (RGB uint8 HxWx3),
        .intrinsics[camera] (3x3), .camera_to_world[camera] (4x4, OpenCV convention) and
        .state. `cameras` is a camera name or a list of them; None means every camera."""
        if isinstance(cameras, str):
            cameras = [cameras]
        data = self.env.call("observe", episode_id=self.episode_id, cameras=cameras)
        self.state = data["state"]
        return Observation(data)

    def show(self, what, caption=""):
        """Attach camera views or an RGB array to the next message you receive."""
        if isinstance(what, Observation):
            for camera, url in what._urls.items():
                self._attachments.append({"caption": f"{caption} {camera}".strip(), "image_url": url})
            return
        import cv2
        import numpy as np

        array = np.asarray(what)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("show() takes an Observation or an HxWx3 RGB array")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("Could not encode the image")
        url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()
        self._attachments.append({"caption": caption, "image_url": url})


def exports(session):
    from harness_policy.runtime.session import ARMS

    return {"finish": session.finish, "__wrap_up__": session._wrap_up, "get_state": session.get_state,
            "get_observation": session.get_observation, "show": session.show, "ARMS": ARMS,
            "CONTROL_HZ": session.hz}
