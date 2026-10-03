"""Version-checked environment and model ports over either RPC transport.

Adapted from the upstream environment/model client separation. Connecting never
implicitly resets a physical scene. Stateful calls have durable request IDs.
"""


class EnvironmentClient:
    def __init__(self, rpc, *, expected):
        self.rpc = rpc
        actual = rpc.call("env.describe")
        if actual != expected:
            raise ValueError(f"Environment identity mismatch: {actual!r}")
        self.last_observation = None

    def observe(self):
        self.last_observation = self.rpc.call("env.observe")
        return self.last_observation

    def reset(self, request_id):
        result = self.rpc.invoke(request_id, "env.reset")
        self.last_observation = result
        return result

    def step(self, request_id, action):
        result = self.rpc.invoke(request_id, "env.step", action=action)
        self.last_observation = result["observation"]
        return result

    def chunk_step(self, request_id, actions, *, return_all_frames=False):
        result = self.rpc.invoke(
            request_id,
            "env.chunk_step",
            actions=actions,
            return_all_frames=return_all_frames,
        )
        self.last_observation = result["observation"]
        return result


class ModelClient:
    def __init__(self, rpc, *, expected):
        self.rpc = rpc
        if rpc.call("model.describe") != expected:
            raise ValueError("Model identity mismatch")

    def predict(self, request_id, observation, options=None):
        options = dict(options or {})
        if "session_id" in options or "session_ids" in options:
            raise ValueError("Model session is owned by the transport")
        return self.rpc.invoke(
            request_id, "model.predict", observation=observation, options=options
        )

    def reset(self, request_id):
        return self.rpc.invoke(request_id, "model.reset")
