"""Real official WS transport, CPU fixture executor; no task score is measured."""
import asyncio
import json
import os
import sys
from unittest.mock import patch

from test_harness import ROOT, FakeBackend, FakeRelease, DIMS, model, contracts, backends, obs
if not os.environ.get("R2SI_TEST_XPL_ROOT"):
    raise RuntimeError("Set R2SI_TEST_XPL_ROOT to an installed XPolicyLab checkout")

from r2si_under_test.server import SingleTrialServer
from client_server.ws.model_server import PolicyServerConfig
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.exceptions import WsError
from XPolicyLab.utils.process_data import encode_image_bit
from websockets.asyncio.server import serve
import cv2
import numpy as np


async def run():
    backend = FakeBackend()
    received = []
    backend.observe = lambda value, previous: received.append(value) or False
    cfg = {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint",
           "policy_name": "WorldScape_RSI", "task_name": "task_a", "eval_batch": False}
    with patch.object(model, "FrozenRelease", return_value=FakeRelease()), \
            patch.object(contracts, "action_dimensions", return_value=DIMS), \
            patch.object(backends, "build_backend", return_value=backend):
        adapter = model.Model(cfg)
        server = SingleTrialServer(adapter, PolicyServerConfig(host="127.0.0.1", port=0))
        async with serve(server._handle_connection, "127.0.0.1", 0) as socket:
            port = socket.sockets[0].getsockname()[1]
            client = PolicyEvalClient(PolicyEvalClientConfig(url=f"ws://127.0.0.1:{port}",
                                      evaluation_id="fixture-evaluation", max_connect_attempts=1,
                                      connect_timeout_s=5, handshake_timeout_s=5, request_timeout_s=5,
                                      close_timeout_s=2))
            await client.connect(handshake=True)
            try:
                await client.reset(trial_id="fixture-a", action_case_id="task_a_case")
                image = np.full((8, 8, 3), [199, 27, 13], dtype=np.uint8)
                legacy = cv2.imencode(".jpg", image)[1].tobytes()
                encodings = [image, encode_image_bit(image), legacy]
                for color in encodings:
                    raw = obs(); raw["vision"]["head_camera"]["color"] = color
                    raw.update(score=1, layout_id=999)
                    await client.call("update_obs", raw, trial_id="fixture-a", action_case_id="task_a_case")
                    frame = await client.call("get_action", None, trial_id="fixture-a", action_case_id="task_a_case")
                    chunk = frame.payload["result"]
                    assert len(chunk) == 1 and set(chunk[0]) == set(DIMS)
                    actual = received[-1]["vision"]["head_camera"]["color"][0,0]
                    assert np.max(np.abs(actual.astype(int)-image[0,0].astype(int))) <= 3
                    assert "score" not in received[-1] and "layout_id" not in received[-1]
                before = adapter._pending.copy()
                try:
                    await client.reset(trial_id="fixture-b", action_case_id="task_b_case")
                except WsError:
                    pass
                else:
                    raise AssertionError("Second trial reset another trial's state")
                assert adapter._pending is not None and adapter._task == "task_a"
                await client.trial_end(trial_id="fixture-a", action_case_id="task_a_case", result={})
                assert server.owner is None and adapter._pending is None
                await client.reset(trial_id="fixture-b", action_case_id="task_b_case")
                await client.trial_end(trial_id="fixture-b", action_case_id="task_b_case", result={})
            finally:
                await client.close()
                adapter.close()
    print(json.dumps({"official_ws_fixture_loop": "passed", "rgb_formats": ["array", "standard", "legacy"],
                      "cross_trial_reset": "rejected", "task_success_measured": False,
                      "real_model_inference": False}))


if __name__ == "__main__":
    asyncio.run(run())
