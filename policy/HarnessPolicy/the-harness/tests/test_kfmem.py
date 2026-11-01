"""The WALL-OSS client keeps head frames and asks only for the frames the server will read."""

import unittest

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
        self.assertEqual([item["sim_frame"] for item in request["ring"]], [10])
        self.assertEqual(request["prev_sim_frame"], None)
        self.assertEqual(request["mem"], [])
        self.assertEqual(state.history(10), [])

    def test_face_image_uses_the_head_camera(self):
        image = _image()
        found = face_image({"vision": {"cam_left_wrist": {"color": _image()}, "cam_head": {"color": image}}})
        self.assertIs(found, image)
