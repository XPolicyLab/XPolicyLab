"""Client-owned frame history for a Wall-X keyframe-memory episode.

The checkpoint selects which past frames to remember by looking at them again, so
the client keeps the raw camera history and ships the frames each request needs.
That is deliberate rather than incidental: the selection pass needs the *pixels*
of its candidate frames (it re-encodes them), and the model server must not
accumulate per-episode video for an evaluation that runs many episodes in
parallel. The first request carries the whole recent window; its response
includes a ``ring_plan``, after which only the frames the server will read are
sent (the current frame, plus the selection candidates when a selection is due).

The state travels as ``kfmem_client_state_v1`` inside the observation. The
server answers with an updated copy (which memory frame ids it retained, how many
selection windows it skipped, the CUDA RNG state it sampled with), and the client
accepts it before the next request. Both sides validate causality — memory ids
must be strictly older than the current frame — so a bug shows up as an explicit
error instead of a silently degraded rollout.
"""

from __future__ import annotations

import re

import numpy as np

#: Frames in the sliding window the server may look at.
RING_FRAMES = 60


def needed_ring_frames(plan, now, last_select_frame):
    """Ring frames the server reads at ``now`` (mirrors the server's arithmetic).

    The current frame, the selection candidates when a selection is due and the
    short-term taps; ``plan`` is the ``ring_plan`` the server returns.
    """
    oldest = max(0, now - int(plan["span"]))
    needed = {now}
    if plan["selection"] and now - last_select_frame >= int(plan["select_stride"]):
        start = max(0, now - int(plan["window"]))
        span = max(1, now - start)
        count = int(plan["num_candidates"])
        needed.update(
            min(max(start + int((i + 0.5) * span / count), oldest), now) for i in range(count)
        )
    taps, stride = (int(v) for v in plan["short_term"])
    needed.update(max(oldest, now - k * stride) for k in range(taps, 0, -1))
    return sorted(needed)


def _ring_plan(value):
    keys = {"span", "window", "num_candidates", "select_stride", "short_term", "selection"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid ring plan")
    if not isinstance(value["selection"], bool) or not (
        isinstance(value["short_term"], (list, tuple)) and len(value["short_term"]) == 2
    ):
        raise ValueError("invalid ring plan")
    numbers = [value[k] for k in ("span", "window", "num_candidates", "select_stride")]
    numbers += list(value["short_term"])
    if any(type(v) is not int or v < 0 for v in numbers) or value["span"] > RING_FRAMES:
        raise ValueError("invalid ring plan")
    return dict(value)


def normalized_box(value):
    """Validate one memory box (four integers in 0-1000, top-left/bottom-right)."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must contain four normalized coordinates")
    if any(type(x) is not int or not 0 <= x <= 1000 for x in value):
        raise ValueError("box coordinates must be integers in 0-1000")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise ValueError("box must have positive width and height")
    return list(value)


class KFMemClientState:
    """Raw frames plus the memory the server reported back, for one episode."""

    def __init__(self, context_key, *, capacity=22):
        if not isinstance(context_key, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", context_key
        ):
            raise ValueError("invalid context key")
        self.context_key = context_key
        self.capacity = capacity
        self.reset(
            {
                "version": 1,
                "prev_sim_frame": None,
                "mem_frame_ids": [],
                "rng_state": None,
            }
        )

    def reset(self, state):
        if (
            state.get("version") != 1
            or state.get("prev_sim_frame") is not None
            or state.get("mem_frame_ids") != []
        ):
            raise ValueError("invalid KFMem reset state")
        self.frames = {}
        self.mem_frame_ids = []
        self.mem_boxes = {}
        self.prev_sim_frame = None
        self.last_sel_frame = self.consec_skip = self.dump_count = 0
        self.rng_state = state.get("rng_state")
        # Set from the first response; until then the full ring is sent.
        self.ring_plan = None

    def record_frame(self, frame, image):
        if type(frame) is not int or frame < 0:
            raise ValueError("frame must be a nonnegative integer")
        image = np.asarray(image)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("frame image must be uint8 HWC3")
        self.frames[frame] = np.ascontiguousarray(image).copy()

    def request_state(self, now):
        """Everything the server needs to answer the request at ``now``."""
        if (
            type(now) is not int
            or now < 0
            or (self.prev_sim_frame is not None and now <= self.prev_sim_frame)
        ):
            raise ValueError("request frame must advance")
        sparse = self.ring_plan is not None
        if sparse:
            # Only the frames the server reads; the model inputs are unchanged.
            ring = needed_ring_frames(self.ring_plan, now, self.last_sel_frame)
        else:
            ring = list(range(max(0, now - RING_FRAMES), now + 1))
        if any(f not in self.frames for f in ring + self.mem_frame_ids):
            raise ValueError("dense raw frames or memory frames are missing")
        state = {
            "version": 1,
            "prev_sim_frame": self.prev_sim_frame,
            "ring": [{"sim_frame": f, "face_view": self.frames[f]} for f in ring],
            "mem": [
                {"sim_frame": f, "face_view": self.frames[f]}
                for f in self.mem_frame_ids
            ],
            "mem_boxes": {str(f): list(box) for f, box in self.mem_boxes.items()},
            "last_sel_frame": self.last_sel_frame,
            "consec_skip": self.consec_skip,
            "dump_count": self.dump_count,
            "rng_state": self.rng_state,
            "context_key": self.context_key,
        }
        if sparse:
            state["ring_sparse"] = True
        return state

    def history(self, now):
        """Frames recorded since the last request (not sent in sparse-ring mode)."""
        if self.ring_plan is not None:
            return []
        first = 1 if self.prev_sim_frame is None else self.prev_sim_frame + 1
        return [
            {"sim_frame": frame, "face_view": self.frames[frame]}
            for frame in range(first, now + 1)
        ]

    def accept_result(self, state):
        """Adopt the memory the server reports after answering at ``now``."""
        if state.get("version") != 1:
            raise ValueError("invalid KFMem state version")
        now = state.get("prev_sim_frame")
        ids = state.get("mem_frame_ids")
        if (
            type(now) is not int
            or now not in self.frames
            or (self.prev_sim_frame is not None and now <= self.prev_sim_frame)
        ):
            raise ValueError("invalid response anchor")
        if (
            not isinstance(ids, list)
            or len(ids) > self.capacity
            or any(type(f) is not int or f not in self.frames or f >= now for f in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("invalid causal memory frame IDs")
        raw_boxes = state.get("mem_boxes", {})
        if not isinstance(raw_boxes, dict):
            raise ValueError("invalid memory box map")
        boxes = {}
        for key, value in raw_boxes.items():
            if (
                not isinstance(key, str)
                or not key.isdecimal()
                or str(int(key)) != key
                or int(key) not in ids
            ):
                raise ValueError("box must reference a retained memory frame")
            boxes[int(key)] = normalized_box(value)
        counters = {}
        for name in ("last_sel_frame", "consec_skip", "dump_count"):
            value = state.get(name)
            if (
                type(value) is not int
                or value < 0
                or (name == "last_sel_frame" and value > now)
            ):
                raise ValueError("invalid KFMem counter: " + name)
            counters[name] = value
        rng = state.get("rng_state")
        if rng is not None and (not isinstance(rng, bytes) or not 0 < len(rng) <= 1 << 20):
            raise ValueError("invalid CUDA RNG state")
        plan = state.get("ring_plan")
        if plan is not None:
            self.ring_plan = _ring_plan(plan)
        self.prev_sim_frame = now
        self.mem_frame_ids = list(ids)
        self.mem_boxes = boxes
        for name, value in counters.items():
            setattr(self, name, value)
        self.rng_state = rng
        keep = set(ids) | set(range(max(0, now - RING_FRAMES), now + 1))
        self.frames = {f: image for f, image in self.frames.items() if f in keep}
