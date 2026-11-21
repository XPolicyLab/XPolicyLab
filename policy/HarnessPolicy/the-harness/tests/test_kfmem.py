"""The WALL-OSS client keeps head frames and asks only for the frames the server will read."""

import unittest
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from skills.vla.kfmem import KFMemClientState, face_image


def _image():
    return np.zeros((4, 4, 3), dtype=np.uint8)


class KFMemTests(unittest.TestCase):
    def test_first_request_carries_the_recorded_frame(self):
        state = KFMemClientState("harness-1")
        state.record_frame(0, _image())
        request = state.request_state(0)
        self.assertEqual(request["ring"][0]["sim_frame"], 0)
        # The first request puts frame 0 in the ring. History starts at frame 1.
        self.assertEqual(state.history(0), [])

    def test_a_gap_continues_from_the_recorded_frame(self):
        state = KFMemClientState("harness-1")
        state.record_frame(0, _image())
        state.request_state(0)
        state.record_frame(10, _image())
        request = state.request_state(10)
        self.assertEqual([item["sim_frame"] for item in request["ring"]], [0])
        self.assertEqual(state.frame_origin, 10)
        self.assertEqual(request["prev_sim_frame"], None)
        self.assertEqual(request["mem"], [])
        self.assertEqual(state.history(10), [])

    def test_no_pixels_are_fabricated_and_restarts_are_reported(self):
        state = KFMemClientState("harness-gap")
        image = _image()
        state.record_frame(100, image)
        request = state.request_state(100)
        self.assertEqual([f["sim_frame"] for f in request["ring"]], [0])
        np.testing.assert_array_equal(request["ring"][0]["face_view"], image)
        self.assertEqual(state.restarts, [{"host_step": 100, "reason": "missing_frames"}])
        state.reset({"version": 1, "prev_sim_frame": None, "mem_frame_ids": []})
        self.assertEqual(state.restarts, [])
        self.assertEqual(state.frame_origin, 0)

    def test_missing_current_frame_still_fails(self):
        state = KFMemClientState("harness-gap")
        with self.assertRaisesRegex(ValueError, "frames are missing"):
            state.request_state(100)


WALL_REPO = Path(__file__).resolve().parents[1] / "third_party/RoboDojo/wall-x"


@unittest.skipUnless((WALL_REPO / "wall_x/serving/robodojo/memory_protocol.py").exists(),
                     "optional WALL checkout needed for real protocol validation (no models)")
class ServerProtocolTests(unittest.TestCase):
    def setUp(self):
        # These server modules are pure frame bookkeeping, with no torch/CUDA.
        sys.path.insert(0, str(WALL_REPO))
        from wall_x.serving.robodojo.memory_protocol import parse_request
        from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
        self.parse = parse_request
        self.config = KeyframeMemoryConfig()
        self.state = KFMemClientState("harness-protocol")

    def tearDown(self):
        sys.path.remove(str(WALL_REPO))

    def request(self, now):
        raw = self.state.request_state(now)
        wire_now = now - self.state.frame_origin
        return self.parse({"sim_frame": wire_now, "kfmem_client_state_v1": raw,
                           "kfmem_history": self.state.history(now)},
                          max_advance=16, config=self.config)

    def accept(self, now, *, memory=()):
        self.state.accept_result({"version": 1, "prev_sim_frame": now,
                                  "mem_frame_ids": list(memory), "last_sel_frame": 0,
                                  "consec_skip": 0, "dump_count": 0, "rng_state": None,
                                  "ring_plan": self.config.ring_plan(""),
                                  "mem_boxes": {str(f): [0, 0, 1000, 1000] for f in memory}})

    def test_late_first_request_and_following_sparse_requests_pass_real_server(self):
        self.state.record_frame(100, _image())
        self.assertEqual(self.request(100).state.now, 0)
        self.accept(0)
        for f in range(101, 117):
            self.state.record_frame(f, _image())
        self.assertEqual(self.request(116).state.now, 16)
        self.accept(16, memory=[0])
        self.assertEqual(self.state.mem_frame_ids, [100])
        self.assertEqual(self.state.mem_boxes, {100: [0, 0, 1000, 1000]})
        for f in range(117, 133):
            self.state.record_frame(f, _image())
        request = self.request(132)
        self.assertEqual(request.previous_frame, 16)
        self.assertEqual([f.index for f in request.state.memory], [0])
        self.assertEqual(request.client_boxes, {0: (0, 0, 1000, 1000)})

    def test_own_tool_gap_restarts_but_does_not_replay_motion(self):
        self.state.record_frame(0, _image())
        self.request(0)
        self.accept(0)
        self.state.record_frame(150, _image())
        self.assertEqual(self.request(150).state.now, 0)
        self.accept(0)
        self.assertEqual(self.state.prev_sim_frame, 150)

    def test_advance_limit_restarts_even_when_required_sparse_pixels_exist(self):
        self.config = replace(self.config, memory_tasks=frozenset())
        self.state.record_frame(0, _image())
        self.request(0)
        self.accept(0)
        self.state.record_frame(100, _image())
        self.assertEqual(self.request(100).state.now, 0)
        self.assertEqual(self.state.restarts[-1]["reason"], "advance_limit")

    def test_face_image_uses_the_head_camera(self):
        image = _image()
        found = face_image({"vision": {"cam_left_wrist": {"color": _image()}, "cam_head": {"color": image}}})
        self.assertIs(found, image)
