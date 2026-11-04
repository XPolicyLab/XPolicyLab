"""Simulation file transport sharing the current durable native adapter."""

import json
import re
from pathlib import Path
import time
from agp.experiments.control_runtime.adapter import RobotAdapter


class RoboDojoAdapter(RobotAdapter):
    dual_arm = True
    supports_estimate = True
    required_cameras = {"top", "left_wrist", "right_wrist"}

    def act(self, action_id, command, args):
        if command != "actions":
            raise ValueError("Simulation requires explicit actions map")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a nonempty string")
        if (not isinstance(args, dict) or set(args) != {"actions"}
                or not isinstance(args["actions"], dict) or not args["actions"]
                or args["actions"].keys() - {"left", "right"}):
            raise ValueError("actions must contain left and/or right")
        for spec in args["actions"].values():
            if (not isinstance(spec, dict) or set(spec) != {"command", "args"}
                    or spec["command"] not in {"move_ee", "move_delta", "move_joints", "home", "gripper", "hold"}
                    or not isinstance(spec["args"], dict)):
                raise ValueError("Invalid arm command envelope")
        payload = json.dumps({"command": command, "args": args}, sort_keys=True, allow_nan=False)
        # Use the existing transport lock, durable reservation and response
        # identity validation unchanged, with ONE reservation for the pair.
        with self._locked(action_id):
            with self._database() as db:
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if row is not None:
                if row["payload"] != payload:
                    raise ValueError("action_id was already used with different arm commands")
                result = self._action_result_locked(row)
                if result is None:
                    raise TimeoutError(f"Action {action_id!r} outcome is unknown; neither arm will be resent")
                return result
            request_id = self._allocate()
            dispatched_ns = time.monotonic_ns()
            with self._database() as db:
                db.execute("INSERT INTO actions (action_id,payload,request_id,dispatched_ns) VALUES (?,?,?,?)",
                           (action_id, payload, request_id, dispatched_ns))
            self._publish(request_id, command, args)
            self._wait(request_id)
            with self._database() as db:
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return self._action_result_locked(row)

    def estimate(self, stages, preview=None, preview_source=None):
        if preview is None:
            return self._call("estimate", {"stages": stages})
        # Only ControlRuntime's saved observation supplies these paths and times.
        source = preview_source
        if (not isinstance(source, dict) or not isinstance(source.get("capture"), str)
                or re.fullmatch(r"[0-9]+", source["capture"]) is None):
            raise ValueError("Invalid preview capture identity")
        camera = preview["camera"]
        if camera not in {"top", "left_wrist", "right_wrist"}:
            raise ValueError("Invalid preview camera")
        def resolve(value):
            path = Path(value)
            path = (path if path.is_absolute() else self.session / path).resolve()
            if not path.is_relative_to(self.session):
                raise ValueError("Preview source must remain inside the session")
            return path

        expected_directory = self.session / "captures" / source["capture"]
        directory = resolve(expected_directory)
        if directory != expected_directory:
            raise ValueError("Preview capture directory cannot redirect through symlinks")
        names = {"rgb_path": camera + ".png", "depth_path": camera + "_depth.npy",
                 "calibration_path": "calibration.json"}
        paths = {key: resolve(directory / name) for key, name in names.items()}
        if any(path != directory / names[key] for key, path in paths.items()):
            raise ValueError("Preview capture files cannot redirect through symlinks")
        selected = [image for image in source["images"] if image.get("label") == camera]
        if len(selected) != 1 or resolve(selected[0]["path"]) != paths["rgb_path"]:
            raise ValueError("Preview RGB differs from the saved observation")
        bound_source = {key: source[key] for key in (
            "observation_id", "revision", "capture", "captured_monotonic_ns", "state")}
        bound_source.update(camera_time_monotonic_ns=source["camera_times"][camera],
                            sim_time_s=source["state"]["sim_time_s"],
                            **{key: str(path) for key, path in paths.items()})
        return self._call("estimate", {"stages": stages, "preview": preview,
                                      "preview_source": bound_source})

    def observe(self):
        return self._call("observe", {})
