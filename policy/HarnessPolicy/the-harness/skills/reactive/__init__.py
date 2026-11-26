"""Controllers that run every control tick (guards come from the reactive layer)."""

from harness_policy.runtime.reactive import ReactiveFault, call_reactive
from harness_policy.runtime.session import CAMERAS


# Consecutive controller calls whose left_eef/right_eef target has no IK solution
# before run_controller stops with REACTIVE_FAULT (the arm holds until then).
CONTROLLER_IK_MISSES = 5
# Near a wrist singularity (the jaws pointing straight down) a small step of a link6
# target can need a large joint motion, which the shield refuses as a jump: the joints of
# an eef target move toward their IK solution by at most this share of the shield's
# joint-jump limit per tick, and the following ticks go on from there.
CONTROLLER_JOINT_STEP = 0.8


class Reactive:
    def run_controller(self, fn, until=None, max_seconds=None, region=None, hz=None, cameras=None,
                       camera_hz=5.0):
        """Run fn(obs) -> command at `hz` (default: every tick) and hold the last command
        in between. The command is a dict with any of 'left_arm'/'right_arm' (6 joint
        targets), 'left_eef'/'right_eef' (link6 pose [x, y, z, qw, qx, qy, qz], solved by
        IK outside fn) and 'left_gripper'/'right_gripper' (0 closed .. 1 open), or None
        to finish. With `cameras`, obs["observation"] is an Observation of those cameras
        refreshed at `camera_hz` (images, intrinsics, camera_to_world), for visual
        servoing with pure image processing and pixel_to_plane / project. fn runs under
        the reactive budget; decoding the images does not count against it."""
        cameras = [cameras] if isinstance(cameras, str) else list(cameras or [])
        unknown = set(cameras) - set(CAMERAS)
        if unknown:
            raise ValueError(f"Unknown cameras {sorted(unknown)}; cameras are {list(CAMERAS)}")
        every = max(1, round(self.hz / float(hz))) if hz else 1
        image_every = max(1, round(self.hz / float(camera_hz))) if cameras else 0
        self._live = self.get_observation(cameras=cameras) if cameras else None
        self._ik_misses = 0
        last = [None]

        def command(state, tick):
            if last[0] is None or tick % every == 0:
                view = self._reactive_view(state, tick)
                if cameras:
                    view["observation"] = self._live
                result = call_reactive("controller", fn, view, budget_seconds=self.reactive.budget_seconds)
                if result is None:
                    return None
                last[0] = self._merge_command(state, result)
            return last[0]

        def cameras_at(tick):
            return cameras if cameras and (tick + 1) % image_every == 0 else None
        return self._execute("run_controller", {"hz": hz, "cameras": cameras}, command, max_seconds,
                             region, until, cameras_at=cameras_at if cameras else None)

    def _merge_command(self, state, result):
        target = self._base_command(state)
        if not isinstance(result, dict):
            raise ReactiveFault("controller", "must return a dict of arm/gripper targets or None")
        for key, value in result.items():
            side, _, part = key.partition("_")
            group = self.groups.get(f"{side}_arm")
            if group is None or part not in ("arm", "gripper", "eef"):
                raise ReactiveFault("controller", f"unknown command key {key!r}")
            if part == "eef":
                pose = [float(v) for v in value]
                if len(pose) != 7:
                    raise ReactiveFault("controller", f"{key} is a link6 pose [x, y, z, qw, qx, qy, qz]")
                value = self.ik(f"{side}_arm", pose)
                if value is None:
                    # A tracked target can leave the reachable space for a moment: hold
                    # the arm, and fault only when it stays unreachable.
                    self._ik_misses = getattr(self, "_ik_misses", 0) + 1
                    if self._ik_misses >= CONTROLLER_IK_MISSES:
                        raise ReactiveFault("controller", f"no IK solution for {key} {[round(v, 3) for v in pose]} "
                                                          f"{self._ik_misses} times in a row")
                    continue
                self._ik_misses = 0
                limit = ((getattr(self, "profile", None) or {}).get("shield") or {}).get("max_joint_jump_rad")
                if limit:
                    step = CONTROLLER_JOINT_STEP * float(limit)
                    now = [float(state["joint_position"][index]) for index in group["joints"]]
                    value = [here + max(-step, min(step, float(number) - here)) for number, here in zip(value, now)]
                part = "arm"
            if part == "arm":
                if len(value) != len(group["joints"]):
                    raise ReactiveFault("controller", f"{key} needs {len(group['joints'])} joint targets")
                for index, number in zip(group["joints"], value):
                    target[index] = float(number)
            else:
                target[group["gripper"]] = float(value)
        return target


def exports(session):
    return {"when": session.reactive.when, "watch": session.reactive.watch, "clear_guards": session.reactive.clear,
            "run_controller": session.run_controller}
