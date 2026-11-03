"""World-action models that predict camera views and motion for an instruction."""

import base64


class Prediction:
    """What a world-action model predicts; see WorldModels.predict_future."""

    def __init__(self, data, horizon=None):
        import cv2
        import numpy as np

        frames = []
        for url in data.get("frames") or []:
            raw = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8)
            frames.append(cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        actions = list(data.get("actions") or [])
        if horizon is not None and actions:
            keep = max(1, min(int(horizon), len(actions)))
            frames = frames[:max(1, round(len(frames) * keep / len(actions)))] if frames else frames
            actions = actions[:keep]
        self.frames, self.actions, self.seconds = frames, actions, data.get("seconds")

    def __repr__(self):
        return f"Prediction(frames={len(self.frames)}, actions={len(self.actions)}, seconds={self.seconds})"


class WorldModels:
    def list_world_models(self):
        """World-action models available to predict_future."""
        return [{"name": name, "description": config.get("description", "")}
                for name, config in self.world_models.items()]

    def predict_future(self, name, instruction, horizon=None):
        """Ask a world-action model what happens if the robot follows `instruction` from
        the current state. Nothing moves. Returns a Prediction with .frames (predicted
        camera images, RGB arrays, oldest first), .actions (predicted end-effector
        targets per 25 Hz step: {"left_ee_pose", "left_ee_joint_state", "right_ee_pose",
        "right_ee_joint_state"}, link6 poses in the env frame, grippers 0 closed .. 1 open)
        and .seconds. horizon keeps the first `horizon` action steps and the frames up to
        that time."""
        import json as _json
        from urllib.request import Request, urlopen

        from harness_policy.services.clients import image_data_url as _data_url

        if name not in self.world_models:
            raise ValueError(f"Unknown world model {name!r}; available: {sorted(self.world_models)}")
        config = self.world_models[name]
        from harness_policy.services.broker import ensure

        ensure(config)  # an on-demand world model is started by the broker first
        obs = self._policy_observation(str(instruction))
        obs["vision"] = {camera: {"color": _data_url(item["color"])} for camera, item in obs["vision"].items()}
        body = _json.dumps({"obs": obs, "seed": int(config.get("seed", 42))}).encode()
        request = Request(config["url"].rstrip("/") + "/predict", data=body,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=float(config.get("timeout_seconds", 900))) as response:
            result = _json.load(response)
        if "error" in result:
            raise RuntimeError(f"{name}: {result['error']}")
        return Prediction(result, horizon)
