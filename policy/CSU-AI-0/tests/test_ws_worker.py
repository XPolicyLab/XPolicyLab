from __future__ import annotations

import importlib
import threading
import unittest


_WsWorker = importlib.import_module(
    "XPolicyLab.policy.CSU-AI-0.expert_proxy"
)._WsWorker


class _FakeClient:
    def __init__(self, **kwargs):
        self.owner = threading.get_ident()
        self.action_case_id = kwargs["action_case_id"]
        self.batch = []

    def call(self, name, obs=None):
        if threading.get_ident() != self.owner:
            raise AssertionError("client used from a foreign thread")
        if name == "update_obs_batch":
            self.batch = list(obs)
            return None
        if name == "get_action_batch":
            return self.batch
        return {"name": name, "obs": obs}

    def close(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("client closed from a foreign thread")


class WsWorkerTest(unittest.TestCase):
    def test_all_calls_stay_on_owner_thread(self):
        worker = _WsWorker(
            url="ws://127.0.0.1:1",
            request_timeout_s=1.0,
            client_factory=_FakeClient,
        )
        worker.wait_ready(1.0)
        self.assertEqual(worker.call("infer", {"x": 1}, 1.0)["name"], "get_action")
        self.assertEqual(worker.call("infer_batch", [{"x": 2}], 1.0), [{"x": 2}])
        worker.close()


if __name__ == "__main__":
    unittest.main()
