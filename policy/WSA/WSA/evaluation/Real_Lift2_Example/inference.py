#!/usr/bin/env python

"""Real Lift2 example robot-side inference session and interaction loop."""

from __future__ import annotations

import atexit
from collections import deque
import os
import queue
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SRC_ROOT = ROOT / "src"

for candidate in [THIS_DIR, ROOT, SRC_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from .remote_client import RealLift2RemoteClient
    from .runtime import (
        Rate,
        robot_action,
    )
except ImportError:
    from remote_client import RealLift2RemoteClient
    from runtime import (
        Rate,
        robot_action,
    )


np.set_printoptions(linewidth=200, suppress=True)
# left:-0.026894,0.009346,0.012016,0.003242,-0.013924,-0.005531,-3.382544,0.002861,0.002861,0.007439,0.008965,-0.002480,-0.009346,-6.758221
# right:-0.000191,-0.000191,0.008202,0.004768,0.002098,-0.008202,-3.382544,0.013924,0.020027,0.020409,0.005531,-0.002861,-0.001335,-6.758602
MANUAL_HOME_INACTIVE = 0
MANUAL_HOME_ACTIVE = 1
MANUAL_HOME_RESUME_GUARD = 2
MANUAL_NUDGE_ACTIONS = {
    "left": np.array(
        [
            -0.026894,
            0.009346,
            0.012016,
            0.003242,
            -0.013924,
            -0.005531,
            -3.382544,
            0.002861,
            0.002861,
            0.007439,
            0.008965,
            -0.002480,
            -0.009346,
            -6.758221,
        ],
        dtype=np.float32,
    ),
    "right": np.array(
        [
            -0.000191,
            -0.000191,
            0.008202,
            0.004768,
            0.002098,
            -0.008202,
            -3.382544,
            0.013924,
            0.020027,
            0.020409,
            0.005531,
            -0.002861,
            -0.001335,
            -6.758602,
        ],
        dtype=np.float32,
    ),
}
DEFAULT_MANUAL_NUDGE_ACTION = MANUAL_NUDGE_ACTIONS["left"]
DEFAULT_MANUAL_NUDGE_NAME = "left"
RIGHT_MANUAL_NUDGE_ACTION = MANUAL_NUDGE_ACTIONS["right"]
RIGHT_MANUAL_NUDGE_NAME = "right"
DEFAULT_MANUAL_NUDGE_HOLD_S = 0.5
_CONSOLE_ORIGINAL_TERMIOS = None
_CONSOLE_KEY_MODE_ENABLED = False


def extract_action_sequence(response, action_dim, keys=("actions", "action")):
    if not isinstance(response, dict):
        return []

    seq = None
    for key in keys:
        if key in response:
            seq = np.asarray(response[key], dtype=np.float32)
            break
    if seq is None:
        return []

    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    elif seq.ndim > 2:
        seq = seq.reshape(seq.shape[0], -1)

    out_seq = []
    for step in seq:
        step = np.asarray(step, dtype=np.float32).reshape(-1)
        if step.size < action_dim:
            out = np.zeros((action_dim,), dtype=np.float32)
            out[: step.size] = step
            out_seq.append(out)
        elif step.size > action_dim:
            out_seq.append(step[:action_dim])
        else:
            out_seq.append(step)
    return out_seq


def read_observation_snapshot(args, shm_dict, shapes):
    obs_dict = {
        "images": {},
        "qpos": None,
        "qvel": None,
        "effort": None,
        "robot_base": None,
        "base_velocity": None,
    }

    for cam in args.camera_names:
        shm, shape, dtype = shm_dict[cam]
        obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
    for state_key in shapes["states"]:
        shm, shape, dtype = shm_dict[state_key]
        obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

    return obs_dict


def compute_prefetch_lead_steps(frame_rate: int, action_seq_len: int, round_trip_ms: float | None, min_lead_steps: int) -> int:
    lead_steps = max(1, int(min_lead_steps))
    if round_trip_ms is not None:
        inferred_steps = int(np.ceil(float(round_trip_ms) * max(1, frame_rate) / 1000.0)) + 1
        lead_steps = max(lead_steps, inferred_steps)
    return min(lead_steps, max(1, action_seq_len - 1))


def print_timing_log(args, *, chunk_idx: int, action_seq_len: int, response: dict | None):
    if args.log_timing_every <= 0:
        return
    if not (chunk_idx <= 3 or chunk_idx % args.log_timing_every == 0):
        return

    server_timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
    client_timing = response.get("client_timing", {}) if isinstance(response, dict) else {}
    infer_ms = server_timing.get("infer_ms")
    pack_ms = client_timing.get("pack_ms")
    round_trip_ms = client_timing.get("round_trip_ms")
    total_client_ms = client_timing.get("total_client_ms")
    payload_bytes = client_timing.get("payload_bytes")
    effective_client_ms = total_client_ms if total_client_ms is not None else round_trip_ms
    chunk_budget_ms = 1000.0 * action_seq_len / max(1, args.frame_rate)

    message = f"[Timing] mode={args.inference_mode} chunk={chunk_idx} horizon={action_seq_len} budget={chunk_budget_ms:.1f}ms"
    if infer_ms is not None:
        message += f" server_infer={float(infer_ms):.1f}ms"
    if pack_ms is not None:
        message += f" pack={float(pack_ms):.1f}ms"
    if round_trip_ms is not None:
        message += f" round_trip={float(round_trip_ms):.1f}ms"
    if total_client_ms is not None:
        message += f" total_client={float(total_client_ms):.1f}ms"
    if effective_client_ms is not None and float(effective_client_ms) > chunk_budget_ms:
        message += "  <- slower than chunk budget, likely to cause stop-go motion"
    if payload_bytes is not None:
        message += f" payload={float(payload_bytes) / (1024.0 * 1024.0):.2f}MB"
    print(message)


def print_chunk_last_action(chunk_idx: int, action_seq) -> None:
    if not action_seq:
        return

    last_action = np.asarray(action_seq[-1], dtype=np.float32).reshape(-1)
    formatted = ", ".join(f"{x:8.4f}" for x in last_action)
    print(
        f"[WSABase] chunk={chunk_idx} received_horizon={len(action_seq)} "
        f"last_step={len(action_seq) - 1} target=[{formatted}]"
    )


def get_response_client_total_ms(response: dict | None) -> float | None:
    if not isinstance(response, dict):
        return None
    client_timing = response.get("client_timing", {})
    if not isinstance(client_timing, dict):
        return None
    total_client_ms = client_timing.get("total_client_ms")
    if total_client_ms is not None:
        return float(total_client_ms)
    round_trip_ms = client_timing.get("round_trip_ms")
    if round_trip_ms is not None:
        return float(round_trip_ms)
    return None


def write_zero_action(shm_dict, action_dim: int, reason: str | None = None) -> np.ndarray:
    action = np.zeros((action_dim,), dtype=np.float32)
    robot_action(action, shm_dict)
    if reason:
        print(reason)
    return action


class LatencyTracker:
    def __init__(self, maxlen: int = 10):
        self._values_ms = deque(maxlen=maxlen)

    def add(self, value_ms: float | None) -> None:
        if value_ms is None:
            return
        self._values_ms.append(float(value_ms))

    def max_ms(self) -> float | None:
        if not self._values_ms:
            return None
        return max(self._values_ms)


class RTCActionQueue:
    def __init__(self):
        self.queue: np.ndarray | None = None
        self.original_queue: np.ndarray | None = None
        self.last_index = 0

    def clear(self) -> None:
        self.queue = None
        self.original_queue = None
        self.last_index = 0

    def qsize(self) -> int:
        if self.queue is None:
            return 0
        return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        return self.last_index

    def get(self) -> np.ndarray | None:
        if self.queue is None or self.last_index >= len(self.queue):
            return None
        action = np.asarray(self.queue[self.last_index], dtype=np.float32).copy()
        self.last_index += 1
        return action

    def get_left_over(self) -> np.ndarray | None:
        if self.original_queue is None:
            return None
        return np.asarray(self.original_queue[self.last_index :], dtype=np.float32).copy()

    def get_processed_left_over(self) -> np.ndarray | None:
        if self.queue is None:
            return None
        return np.asarray(self.queue[self.last_index :], dtype=np.float32).copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> None:
        reported_delay = max(0, int(real_delay))
        effective_delay = reported_delay
        if action_index_before_inference is not None and self.queue is not None:
            indexes_diff = max(0, self.last_index - int(action_index_before_inference))
            if indexes_diff != reported_delay:
                print(
                    "[WSABase] RTC queue observed consumed steps do not match reported delay: "
                    f"observed={indexes_diff}, reported={reported_delay}. "
                    "Using the observed queue consumption for prefix alignment."
                )
            effective_delay = indexes_diff

        clamped_delay = max(0, min(effective_delay, len(original_actions), len(processed_actions)))
        self.original_queue = np.asarray(original_actions[clamped_delay:], dtype=np.float32).copy()
        self.queue = np.asarray(processed_actions[clamped_delay:], dtype=np.float32).copy()
        self.last_index = 0


def set_manual_home_command(manual_home_command, enabled: bool) -> None:
    set_manual_home_state(manual_home_command, MANUAL_HOME_ACTIVE if enabled else MANUAL_HOME_INACTIVE)


def set_manual_home_state(manual_home_command, state: int) -> None:
    with manual_home_command.get_lock():
        manual_home_command.value = int(state)


def get_manual_home_state(manual_home_command) -> int:
    with manual_home_command.get_lock():
        return int(manual_home_command.value)


def wait_for_manual_home_resume_guard(args, ros_proc, manual_home_command) -> bool:
    """Wait until the ROS process finishes publishing zero actions after resume."""
    expected_guard_s = max(0, int(args.manual_home_resume_guard_steps)) / max(1, int(args.frame_rate))
    timeout_s = max(2.0, expected_guard_s + 1.0)
    start_time = time.monotonic()
    timeout_warning_emitted = False
    while ros_proc.is_alive():
        if get_manual_home_state(manual_home_command) == MANUAL_HOME_INACTIVE:
            return True
        if not timeout_warning_emitted and time.monotonic() - start_time > timeout_s:
            print(
                "[Manual Home] Resume guard did not finish before timeout; "
                "still waiting so fresh inference does not overwrite the zero-action flush."
            )
            timeout_warning_emitted = True
        time.sleep(0.02)
    return False


def print_manual_home_help() -> None:
    print(
        "[Manual Home] Controls:\n"
        "  - Press Enter once: home both arms and pause the current rollout.\n"
        "  - Press Enter again while paused: start a brand-new rollout from timestep 0 after a brief zero-action flush.\n"
        "    That second Enter will also count as the first-chunk confirmation for the restarted rollout.\n"
        "  - Press I: publish the left manual nudge pose for 0.5s, then request fresh inference.\n"
        "  - Press O: publish the right manual nudge pose for 0.5s, then request fresh inference.\n"
        "  - Base height will stay unchanged during manual home if fixed height is enabled.\n"
    )


def restore_console_key_mode() -> None:
    global _CONSOLE_ORIGINAL_TERMIOS, _CONSOLE_KEY_MODE_ENABLED
    if not _CONSOLE_KEY_MODE_ENABLED or _CONSOLE_ORIGINAL_TERMIOS is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _CONSOLE_ORIGINAL_TERMIOS)
    except Exception:
        pass
    finally:
        _CONSOLE_ORIGINAL_TERMIOS = None
        _CONSOLE_KEY_MODE_ENABLED = False


def enable_console_key_mode() -> None:
    global _CONSOLE_ORIGINAL_TERMIOS, _CONSOLE_KEY_MODE_ENABLED
    if _CONSOLE_KEY_MODE_ENABLED or not sys.stdin or not sys.stdin.isatty() or os.name != "posix":
        return
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        _CONSOLE_ORIGINAL_TERMIOS = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        _CONSOLE_KEY_MODE_ENABLED = True
        atexit.register(restore_console_key_mode)
        print("[Manual Controls] Single-key input enabled: Enter=home/resume, I=left nudge, O=right nudge.")
    except Exception as exc:
        print(f"[Manual Controls] Single-key input unavailable; use Enter, 'i'+Enter, or 'o'+Enter instead: {exc}")


def poll_manual_console_command(manual_home_active: bool) -> str | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None

    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return None

    if not ready:
        return None

    if _CONSOLE_KEY_MODE_ENABLED:
        try:
            char = sys.stdin.read(1)
        except Exception:
            return None

        if char in {"\n", "\r"}:
            return "resume" if manual_home_active else "home"
        command = char.strip().lower()
        if command == "i":
            return "nudge_left"
        if command == "o":
            return "nudge_right"
        if command == "h" and manual_home_active:
            return "resume"
        return command or None

    try:
        line = sys.stdin.readline()
    except Exception:
        return None
    if line == "":
        return None

    command = line.strip().lower()
    if command == "":
        if manual_home_active:
            return "resume"
        return "home"
    if command == "i":
        return "nudge_left"
    if command == "o":
        return "nudge_right"
    if manual_home_active and command == "h":
        return "resume"
    return command or None


def build_manual_nudge_action(action_dim: int, nudge_name: str = DEFAULT_MANUAL_NUDGE_NAME) -> np.ndarray:
    target = MANUAL_NUDGE_ACTIONS.get(nudge_name)
    if target is None:
        raise ValueError(f"Unknown manual nudge target: {nudge_name!r}")

    action = np.zeros((action_dim,), dtype=np.float32)
    usable_dims = min(action_dim, target.shape[0])
    action[:usable_dims] = target[:usable_dims]
    return action


def publish_manual_nudge(args, shm_dict, action_dim: int, nudge_name: str = DEFAULT_MANUAL_NUDGE_NAME) -> None:
    action = build_manual_nudge_action(action_dim, nudge_name)
    key_name = "O" if nudge_name == RIGHT_MANUAL_NUDGE_NAME else "I"
    formatted = ", ".join(f"{x:8.4f}" for x in action)
    print(
        "\n"
        + "=" * 72
        + f"\n[Manual Nudge] {key_name} detected. Publishing {nudge_name} manual nudge target for "
        f"{DEFAULT_MANUAL_NUDGE_HOLD_S:.2f}s before requesting fresh inference.\n"
        f"[Manual Nudge] target=[{formatted}]"
        + "\n"
        + "=" * 72
        + "\n"
    )
    robot_action(action, shm_dict)
    time.sleep(DEFAULT_MANUAL_NUDGE_HOLD_S)


def maybe_handle_manual_command(args, ros_proc, shm_dict, manual_home_command, action_dim: int) -> tuple[bool, bool, bool]:
    command = poll_manual_console_command(manual_home_active=False)
    if command in {"nudge", "nudge_left", "nudge_right"}:
        nudge_name = RIGHT_MANUAL_NUDGE_NAME if command == "nudge_right" else DEFAULT_MANUAL_NUDGE_NAME
        publish_manual_nudge(args, shm_dict, action_dim, nudge_name)
        return False, False, True
    if command != "home":
        return False, False, False

    set_manual_home_command(manual_home_command, True)
    robot_action(np.zeros((action_dim,), dtype=np.float32), shm_dict)
    print(
        "\n"
        + "=" * 72
        + "\n[Manual Home] Enter detected. Homing both arms now while keeping the base height unchanged.\n"
        "[Manual Home] CURRENT ROLLOUT IS PAUSED.\n"
        "[Manual Home] After the arms settle and you finish resetting the scene,\n"
        "[Manual Home] PRESS ENTER AGAIN TO START A BRAND-NEW ROLLOUT FROM TIMESTEP 0.\n"
        "[Manual Home] After that second Enter, the robot will briefly hold zero actions,\n"
        "[Manual Home] and that same Enter will also count as the first-chunk safety confirmation.\n"
        + "=" * 72
        + "\n"
    )

    last_resume_reminder_time = 0.0
    while ros_proc.is_alive():
        now = time.monotonic()
        if now - last_resume_reminder_time >= 1.5:
            print(
                "[Manual Home] Paused at home pose. "
                "Press Enter again to start a fresh rollout from timestep 0."
            )
            last_resume_reminder_time = now
        resume_command = poll_manual_console_command(manual_home_active=True)
        if resume_command == "resume":
            robot_action(np.zeros((action_dim,), dtype=np.float32), shm_dict)
            set_manual_home_state(manual_home_command, MANUAL_HOME_RESUME_GUARD)
            print(
                "[Manual Home] Second Enter detected.\n"
                "[Manual Home] Leaving home-pause state and briefly flushing stale actions before fresh inference.\n"
                "[Manual Home] This second Enter will also be reused as the first-chunk confirmation.\n"
            )
            wait_for_manual_home_resume_guard(args, ros_proc, manual_home_command)
            return True, True, False
        if resume_command not in (None, "home"):
            print(
                "[Manual Home] Still paused at home pose. "
                "Press Enter again to start a fresh rollout from timestep 0."
            )
        time.sleep(0.05)

    set_manual_home_command(manual_home_command, False)
    return True, False, False


def maybe_run_first_safety_check(args, response, obs_dict, action_dim, *, auto_confirm: bool = False):
    qpos = np.asarray(
        obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)), dtype=np.float32
    ).reshape(-1)
    qpos_full = np.zeros((args.state_dim,), dtype=np.float32)
    qpos_full[: min(args.state_dim, qpos.size)] = qpos[: args.state_dim]

    if not isinstance(response, dict) or response.get("actions", None) is None:
        print("[SafeGuard] WSABase response is missing `actions`, aborting before robot execution.")
        return False

    actions_seq = np.asarray(response["actions"], dtype=np.float32)
    if actions_seq.ndim != 2 or actions_seq.shape[1] != action_dim:
        print(
            f"[SafeGuard] WSABase returned unexpected action shape: {actions_seq.shape}, "
            f"expected (N, {action_dim})."
        )
        return False

    print("\n" + "=" * 72)
    print("[First Safety Check] Received first WSABase action chunk, pausing before execution")
    print("=" * 72)
    print("Current qpos:")
    print("[" + ", ".join([f"{x:8.4f}" for x in qpos_full]) + "]")

    print("\nPredicted actions (first up to 30 steps):")
    num_steps = min(30, actions_seq.shape[0])
    for idx in range(num_steps):
        print(f"Step {idx:2d}: [" + ", ".join([f"{x:8.4f}" for x in actions_seq[idx]]) + "]")

    if auto_confirm:
        print(
            "\n[Safety Confirm] Reusing the second manual-home Enter as the "
            "confirmation for this restarted rollout."
        )
        print("[Safety Confirmed] Starting normal real-time execution.\n")
        return True

    user_input = input(
        "\n[Safety Confirm] Inspect the actions and press Enter to continue "
        "(input any other text to abort): "
    ).strip().lower()
    if user_input not in {"", "y"}:
        print("[SafeGuard] User aborted during the first safety check, exiting without robot execution.")
        return False

    print("[Safety Confirmed] Starting normal real-time execution.\n")
    return True


class AsyncChunkPrefetcher:
    """Background websocket inference worker that can prefetch the next action chunk."""

    def __init__(self, build_client, *, max_history: int):
        self._build_client = build_client
        self._request_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._pending_observations = deque(maxlen=max(1, int(max_history)))
        self._inflight = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        client = self._build_client()
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if task is None:
                    return

                try:
                    for history_images in task.get("history_observations", []):
                        client.observe(history_images)
                    response = client.infer_step(
                        images=task["images"],
                        qpos=task["qpos"],
                        timestep=task["timestep"],
                        prompt=task["prompt"],
                        inference_delay=task.get("inference_delay"),
                        prev_chunk_left_over=task.get("prev_chunk_left_over"),
                        prev_chunk_left_over_processed=task.get("prev_chunk_left_over_processed"),
                        update_history=False,
                    )
                    result = {
                        "ok": True,
                        "response": response,
                        "timestep": task["timestep"],
                        "action_index_before_inference": task.get("action_index_before_inference"),
                    }
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": exc,
                        "timestep": task["timestep"],
                        "action_index_before_inference": task.get("action_index_before_inference"),
                    }
                    try:
                        client.close()
                    except Exception:
                        pass
                    try:
                        client = self._build_client()
                        client.reset()
                    except Exception as reconnect_exc:
                        result["reconnect_error"] = reconnect_exc

                while not self._stop_event.is_set():
                    try:
                        self._result_queue.put(result, timeout=0.1)
                        break
                    except queue.Full:
                        try:
                            self._result_queue.get_nowait()
                        except queue.Empty:
                            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def observe(self, images) -> None:
        snapshot = {
            camera_name: np.asarray(image).copy()
            for camera_name, image in images.items()
        }
        with self._history_lock:
            self._pending_observations.append(snapshot)

    def _snapshot_pending_observations(self) -> list[dict[str, np.ndarray]]:
        with self._history_lock:
            observations = list(self._pending_observations)
            self._pending_observations.clear()
        return observations

    def _restore_pending_observations(self, observations: list[dict[str, np.ndarray]]) -> None:
        if not observations:
            return
        with self._history_lock:
            restored = list(observations) + list(self._pending_observations)
            self._pending_observations.clear()
            self._pending_observations.extend(restored[-self._pending_observations.maxlen :])

    def submit(
        self,
        *,
        images,
        qpos,
        timestep: int,
        prompt: str,
        inference_delay: int | None = None,
        prev_chunk_left_over=None,
        prev_chunk_left_over_processed=None,
        action_index_before_inference: int | None = None,
    ) -> bool:
        self.observe(images)
        with self._lock:
            if self._inflight or self._stop_event.is_set():
                return False
            self._inflight = True

        history_observations = self._snapshot_pending_observations()
        task = {
            "images": images,
            "qpos": np.asarray(qpos, dtype=np.float32).reshape(-1),
            "timestep": int(timestep),
            "prompt": prompt,
            "action_index_before_inference": action_index_before_inference,
            "history_observations": history_observations,
        }
        if inference_delay is not None:
            task["inference_delay"] = int(inference_delay)
        if prev_chunk_left_over is not None:
            task["prev_chunk_left_over"] = np.asarray(prev_chunk_left_over, dtype=np.float32).copy()
        if prev_chunk_left_over_processed is not None:
            task["prev_chunk_left_over_processed"] = np.asarray(
                prev_chunk_left_over_processed,
                dtype=np.float32,
            ).copy()

        try:
            self._request_queue.put_nowait(task)
            return True
        except queue.Full:
            self._restore_pending_observations(history_observations)
            with self._lock:
                self._inflight = False
            return False

    def has_inflight(self) -> bool:
        with self._lock:
            return self._inflight

    def poll(self):
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._inflight = False
        return result

    def wait(self, timeout: float | None = None):
        try:
            result = self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            self._inflight = False
        return result

    def stop(self):
        self._stop_event.set()
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)


def update_image_history(history_target, args, shm_dict, shapes, *, warning_state: dict[str, bool]) -> None:
    """Keep image history in control-frame units, not request/chunk units."""
    try:
        obs_dict = read_observation_snapshot(args, shm_dict, shapes)
        history_target.observe(obs_dict["images"])
    except Exception as exc:
        if not warning_state.get("emitted", False):
            print(f"[WSABase] Failed to update image history during chunk execution: {exc}")
            warning_state["emitted"] = True


def inference_process(args, config, shm_dict, shapes, ros_proc, manual_home_command):
    ws_url = args.ws_url or os.getenv("REAL_LIFT2_WS_URL", "ws://127.0.0.1:8000")
    action_dim = config["policy_config"]["action_dim"]
    send_image_height = args.send_image_height if args.send_image_height > 0 else None
    send_image_width = args.send_image_width if args.send_image_width > 0 else None
    action = np.zeros((action_dim,), dtype=np.float32)
    robot_action(action, shm_dict)
    first_inference = True
    exec_rate = Rate(args.frame_rate)
    chunk_idx = 0

    def build_client():
        return RealLift2RemoteClient(
            host=ws_url,
            prompt=args.prompt,
            image_history_interval=args.image_history_interval,
            state_dim=args.state_dim,
            max_history=args.image_history_interval + 1,
            send_image_height=send_image_height,
            send_image_width=send_image_width,
        )

    metadata_client = build_client()
    server_metadata = {}
    print(f"[WSABase] WebSocket inference enabled: {ws_url}")
    if send_image_height is not None and send_image_width is not None:
        print(
            f"[WSABase] Robot-side request images will be downsampled to "
            f"{send_image_height}x{send_image_width} before websocket transfer."
        )
    else:
        print("[WSABase] Robot-side request images use the native camera resolution.")
    try:
        server_metadata = metadata_client.metadata
        print(f"[WSABase] server metadata keys: {list(server_metadata.keys())}")
    except Exception as exc:
        if args.inference_mode == "rtc":
            raise RuntimeError(
                "RTC mode requires readable server metadata so the robot can verify "
                f"rtc/action_mode compatibility before execution. Metadata fetch failed: {exc}"
            ) from exc
    finally:
        metadata_client.close()

    rtc_action_mode = str(server_metadata.get("action_mode", "abs")).lower()
    if args.inference_mode == "rtc":
        if not server_metadata.get("rtc_enabled", False):
            raise RuntimeError(
                "Robot-side rtc mode was requested, but the server metadata does not report rtc_enabled=true. "
                "Start the GPU serve with RTC_ENABLED=true or switch the robot back to sync/async mode."
            )
        if rtc_action_mode not in {"abs", "delta"}:
            raise RuntimeError(
                "RTC mode requires the server metadata to expose action_mode as 'abs' or 'delta', "
                f"got {server_metadata.get('action_mode')!r}."
            )
        server_execution_horizon = server_metadata.get("rtc_execution_horizon")
        if server_execution_horizon is not None and int(server_execution_horizon) != int(args.rtc_execution_horizon):
            print(
                "[WSABase] Warning: robot/server RTC execution horizon mismatch: "
                f"robot={args.rtc_execution_horizon}, server={server_execution_horizon}."
            )
        print(
            "[WSABase] Server-side RTC is enabled with "
            f"execution_horizon={server_metadata.get('rtc_execution_horizon')} "
            f"and schedule={server_metadata.get('rtc_prefix_attention_schedule')}."
        )
        if rtc_action_mode == "delta":
            print(
                "[WSABase] RTC will send absolute leftover actions back to the server for "
                "delta-action re-anchoring."
            )
        else:
            print("[WSABase] RTC will reuse model-space leftover actions directly.")
    print_manual_home_help()
    enable_console_key_mode()
    auto_confirm_next_first_chunk = False

    while ros_proc.is_alive():
        timestep = 0
        episode_restart_requested = False
        if args.inference_mode == "async":
            prefetcher = AsyncChunkPrefetcher(build_client, max_history=args.image_history_interval + 1)
            current_response = None
            current_obs = None
            next_response = None
            last_round_trip_ms = None
            async_history_warning = {"emitted": False}

            try:
                while timestep < args.max_publish_step and ros_proc.is_alive():
                    manual_home_restart, manual_home_auto_confirm, manual_nudge = maybe_handle_manual_command(
                        args,
                        ros_proc,
                        shm_dict,
                        manual_home_command,
                        action_dim,
                    )
                    if manual_home_restart:
                        episode_restart_requested = True
                        auto_confirm_next_first_chunk = manual_home_auto_confirm
                        break
                    if manual_nudge:
                        current_response = None
                        current_obs = None
                        next_response = None
                        prefetcher.stop()
                        prefetcher = AsyncChunkPrefetcher(
                            build_client,
                            max_history=args.image_history_interval + 1,
                        )

                    if current_response is None:
                        if next_response is not None:
                            current_response = next_response
                            next_response = None
                        else:
                            if not prefetcher.has_inflight():
                                current_obs = read_observation_snapshot(args, shm_dict, shapes)
                                submitted = prefetcher.submit(
                                    images=current_obs["images"],
                                    qpos=current_obs.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                                    timestep=timestep,
                                    prompt=args.prompt,
                                )
                                if not submitted:
                                    time.sleep(0.002)
                                    continue

                            result = prefetcher.wait(timeout=5.0)
                            if result is None:
                                action = write_zero_action(
                                    shm_dict,
                                    action_dim,
                                    "[SafeGuard] Waiting for prefetched chunk timed out, forcing zero action for this cycle.",
                                )
                                continue

                            if not result.get("ok", False):
                                action = write_zero_action(
                                    shm_dict,
                                    action_dim,
                                    f"[SafeGuard] Remote inference failed, forcing zero action for this cycle: {result['error']}",
                                )
                                reconnect_exc = result.get("reconnect_error")
                                if reconnect_exc is not None:
                                    print(f"[WSABase] reconnect failed: {reconnect_exc}")
                                timestep += 1
                                continue

                            current_response = result.get("response")

                    response = current_response
                    current_response = None

                    if response is None:
                        action = write_zero_action(
                            shm_dict,
                            action_dim,
                            "[SafeGuard] WSABase server returned nothing, forcing zero action for this cycle.",
                        )
                        timestep += 1
                        continue

                    action_seq = extract_action_sequence(response, action_dim)
                    if len(action_seq) == 0:
                        action = write_zero_action(
                            shm_dict,
                            action_dim,
                            "[SafeGuard] WSABase returned an empty action sequence, forcing zero action for this cycle.",
                        )
                        timestep += 1
                        continue

                    if first_inference:
                        if current_obs is None:
                            current_obs = read_observation_snapshot(args, shm_dict, shapes)
                        if not maybe_run_first_safety_check(
                            args,
                            response,
                            current_obs,
                            action_dim,
                            auto_confirm=auto_confirm_next_first_chunk,
                        ):
                            write_zero_action(shm_dict, action_dim, "[SafeGuard] First safety check failed, zeroing action buffer.")
                            return
                        first_inference = False
                        auto_confirm_next_first_chunk = False

                    chunk_idx += 1
                    if isinstance(response, dict):
                        client_timing = response.get("client_timing", {})
                        if client_timing is not None:
                            last_round_trip_ms = client_timing.get(
                                "total_client_ms",
                                client_timing.get("round_trip_ms", last_round_trip_ms),
                            )

                    print_timing_log(args, chunk_idx=chunk_idx, action_seq_len=len(action_seq), response=response)
                    print_chunk_last_action(chunk_idx, action_seq)

                    prefetch_lead_steps = compute_prefetch_lead_steps(
                        frame_rate=args.frame_rate,
                        action_seq_len=len(action_seq),
                        round_trip_ms=last_round_trip_ms,
                        min_lead_steps=args.prefetch_lead_steps,
                    )

                    for step_idx, step_action in enumerate(action_seq):
                        if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                            break

                        history_updated_this_step = False
                        manual_home_restart, manual_home_auto_confirm, manual_nudge = maybe_handle_manual_command(
                            args,
                            ros_proc,
                            shm_dict,
                            manual_home_command,
                            action_dim,
                        )
                        if manual_home_restart:
                            episode_restart_requested = True
                            auto_confirm_next_first_chunk = manual_home_auto_confirm
                            break
                        if manual_nudge:
                            current_response = None
                            current_obs = None
                            next_response = None
                            prefetcher.stop()
                            prefetcher = AsyncChunkPrefetcher(
                                build_client,
                                max_history=args.image_history_interval + 1,
                            )
                            break

                        remaining_steps = len(action_seq) - step_idx - 1
                        if (
                            next_response is None
                            and not prefetcher.has_inflight()
                            and remaining_steps <= prefetch_lead_steps
                            and timestep + 1 < args.max_publish_step
                        ):
                            next_obs = read_observation_snapshot(args, shm_dict, shapes)
                            prefetcher.submit(
                                images=next_obs["images"],
                                qpos=next_obs.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                                timestep=timestep,
                                prompt=args.prompt,
                            )
                            history_updated_this_step = True

                        maybe_result = prefetcher.poll()
                        if maybe_result is not None:
                            if maybe_result.get("ok", False):
                                next_response = maybe_result.get("response")
                            else:
                                print(f"[WSABase] background prefetch failed: {maybe_result['error']}")
                                reconnect_exc = maybe_result.get("reconnect_error")
                                if reconnect_exc is not None:
                                    print(f"[WSABase] reconnect failed: {reconnect_exc}")

                        if not history_updated_this_step:
                            update_image_history(
                                prefetcher,
                                args,
                                shm_dict,
                                shapes,
                                warning_state=async_history_warning,
                            )
                        action = step_action
                        robot_action(action, shm_dict)
                        timestep += 1
                        exec_rate.sleep()

                    if episode_restart_requested:
                        break
            finally:
                prefetcher.stop()
        elif args.inference_mode == "rtc":
            prefetcher = AsyncChunkPrefetcher(build_client, max_history=args.image_history_interval + 1)
            latency_tracker = LatencyTracker(maxlen=args.rtc_latency_lookback)
            action_queue = RTCActionQueue()
            pending_obs = None
            rtc_queue_warning_emitted = False
            rtc_history_warning = {"emitted": False}

            try:
                while timestep < args.max_publish_step and ros_proc.is_alive():
                    history_updated_this_step = False
                    manual_home_restart, manual_home_auto_confirm, manual_nudge = maybe_handle_manual_command(
                        args,
                        ros_proc,
                        shm_dict,
                        manual_home_command,
                        action_dim,
                    )
                    if manual_home_restart:
                        episode_restart_requested = True
                        auto_confirm_next_first_chunk = manual_home_auto_confirm
                        break
                    if manual_nudge:
                        action_queue.clear()
                        pending_obs = None
                        prefetcher.stop()
                        prefetcher = AsyncChunkPrefetcher(
                            build_client,
                            max_history=args.image_history_interval + 1,
                        )
                        continue

                    if action_queue.qsize() <= args.rtc_queue_threshold and not prefetcher.has_inflight():
                        pending_obs = read_observation_snapshot(args, shm_dict, shapes)
                        prev_chunk_left_over = None
                        prev_chunk_left_over_processed = None
                        if rtc_action_mode == "delta":
                            prev_chunk_left_over_processed = action_queue.get_processed_left_over()
                        else:
                            prev_chunk_left_over = action_queue.get_left_over()
                        estimated_client_ms = latency_tracker.max_ms()
                        inference_delay = 0
                        if estimated_client_ms is not None:
                            inference_delay = int(np.ceil(estimated_client_ms * max(1, args.frame_rate) / 1000.0))

                        submitted = prefetcher.submit(
                            images=pending_obs["images"],
                            qpos=pending_obs.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                            timestep=timestep,
                            prompt=args.prompt,
                            inference_delay=inference_delay,
                            prev_chunk_left_over=prev_chunk_left_over,
                            prev_chunk_left_over_processed=prev_chunk_left_over_processed,
                            action_index_before_inference=action_queue.get_action_index(),
                        )
                        history_updated_this_step = True
                        if not submitted:
                            pending_obs = None

                    maybe_result = prefetcher.poll()
                    if maybe_result is not None:
                        if not maybe_result.get("ok", False):
                            print(f"[WSABase] rtc background inference failed: {maybe_result['error']}")
                            reconnect_exc = maybe_result.get("reconnect_error")
                            if reconnect_exc is not None:
                                print(f"[WSABase] reconnect failed: {reconnect_exc}")
                        else:
                            response = maybe_result.get("response")
                            if response is None:
                                write_zero_action(
                                    shm_dict,
                                    action_dim,
                                    "[SafeGuard] RTC request returned nothing, keeping the robot stopped until a fresh chunk arrives.",
                                )
                            else:
                                action_seq = extract_action_sequence(response, action_dim)
                                model_action_seq = extract_action_sequence(
                                    response,
                                    action_dim,
                                    keys=("model_actions", "model_action"),
                                )

                                if len(action_seq) == 0 or len(model_action_seq) == 0:
                                    print(
                                        "[SafeGuard] RTC response is missing processed actions or model_actions; "
                                        "keeping the queue unchanged."
                                    )
                                elif len(action_seq) != len(model_action_seq):
                                    print(
                                        "[SafeGuard] RTC response returned mismatched processed/model action lengths; "
                                        "keeping the queue unchanged."
                                    )
                                else:
                                    chunk_idx += 1
                                    print_timing_log(
                                        args,
                                        chunk_idx=chunk_idx,
                                        action_seq_len=len(action_seq),
                                        response=response,
                                    )
                                    print_chunk_last_action(chunk_idx, action_seq)

                                    client_total_ms = get_response_client_total_ms(response)
                                    latency_tracker.add(client_total_ms)
                                    real_delay = 0
                                    if client_total_ms is not None:
                                        real_delay = int(
                                            np.ceil(float(client_total_ms) * max(1, args.frame_rate) / 1000.0)
                                        )

                                    if (
                                        args.rtc_queue_threshold < args.rtc_execution_horizon + real_delay
                                        and not rtc_queue_warning_emitted
                                    ):
                                        print(
                                            "[WSABase] rtc_queue_threshold looks too small for the observed latency. "
                                            f"threshold={args.rtc_queue_threshold}, "
                                            f"execution_horizon={args.rtc_execution_horizon}, "
                                            f"real_delay={real_delay}."
                                        )
                                        rtc_queue_warning_emitted = True

                                    processed_actions = np.stack(action_seq).astype(np.float32, copy=False)
                                    model_actions = np.stack(model_action_seq).astype(np.float32, copy=False)

                                    if first_inference:
                                        remaining_actions = processed_actions[real_delay:]
                                        if remaining_actions.size == 0:
                                            print(
                                                "[SafeGuard] First RTC chunk became stale before execution. "
                                                "Requesting another chunk before moving the robot."
                                            )
                                        else:
                                            safety_response = dict(response)
                                            safety_response["actions"] = remaining_actions
                                            safety_obs = (
                                                pending_obs
                                                if pending_obs is not None
                                                else read_observation_snapshot(args, shm_dict, shapes)
                                            )
                                            if not maybe_run_first_safety_check(
                                                args,
                                                safety_response,
                                                safety_obs,
                                                action_dim,
                                                auto_confirm=auto_confirm_next_first_chunk,
                                            ):
                                                write_zero_action(
                                                    shm_dict,
                                                    action_dim,
                                                    "[SafeGuard] First safety check failed, zeroing action buffer.",
                                                )
                                                return
                                            first_inference = False
                                            auto_confirm_next_first_chunk = False

                                    action_queue.merge(
                                        model_actions,
                                        processed_actions,
                                        real_delay,
                                        maybe_result.get("action_index_before_inference"),
                                    )
                        pending_obs = None

                    if not history_updated_this_step:
                        update_image_history(
                            prefetcher,
                            args,
                            shm_dict,
                            shapes,
                            warning_state=rtc_history_warning,
                        )
                    next_action = action_queue.get()
                    if next_action is None:
                        action = write_zero_action(shm_dict, action_dim)
                    else:
                        action = np.asarray(next_action, dtype=np.float32)
                        robot_action(action, shm_dict)

                    timestep += 1
                    exec_rate.sleep()
            finally:
                prefetcher.stop()
        else:
            client = build_client()
            client.reset()
            sync_history_warning = {"emitted": False}
            while timestep < args.max_publish_step and ros_proc.is_alive():
                manual_home_restart, manual_home_auto_confirm, manual_nudge = maybe_handle_manual_command(
                    args,
                    ros_proc,
                    shm_dict,
                    manual_home_command,
                    action_dim,
                )
                if manual_home_restart:
                    episode_restart_requested = True
                    auto_confirm_next_first_chunk = manual_home_auto_confirm
                    break
                if manual_nudge:
                    client.reset()

                obs_dict = read_observation_snapshot(args, shm_dict, shapes)

                try:
                    response = client.infer_step(
                        images=obs_dict["images"],
                        qpos=obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                        timestep=timestep,
                        prompt=args.prompt,
                    )
                except Exception as exc:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        f"[SafeGuard] Remote inference failed, forcing zero action for this cycle: {exc}",
                    )
                    try:
                        client.close()
                    except Exception:
                        pass
                    try:
                        client = build_client()
                        client.reset()
                    except Exception as reconnect_exc:
                        print(f"[WSABase] reconnect failed: {reconnect_exc}")
                    timestep += 1
                    continue

                if response is None:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        "[SafeGuard] WSABase server returned nothing, forcing zero action for this cycle.",
                    )
                    timestep += 1
                    continue

                action_seq = extract_action_sequence(response, action_dim)
                if len(action_seq) == 0:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        "[SafeGuard] WSABase returned an empty action sequence, forcing zero action for this cycle.",
                    )
                    timestep += 1
                    continue

                if first_inference:
                    if not maybe_run_first_safety_check(
                        args,
                        response,
                        obs_dict,
                        action_dim,
                        auto_confirm=auto_confirm_next_first_chunk,
                    ):
                        write_zero_action(shm_dict, action_dim, "[SafeGuard] First safety check failed, zeroing action buffer.")
                        return
                    first_inference = False
                    auto_confirm_next_first_chunk = False

                chunk_idx += 1
                print_timing_log(args, chunk_idx=chunk_idx, action_seq_len=len(action_seq), response=response)
                print_chunk_last_action(chunk_idx, action_seq)

                for step_action in action_seq:
                    if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                        break

                    manual_home_restart, manual_home_auto_confirm, manual_nudge = maybe_handle_manual_command(
                        args,
                        ros_proc,
                        shm_dict,
                        manual_home_command,
                        action_dim,
                    )
                    if manual_home_restart:
                        episode_restart_requested = True
                        auto_confirm_next_first_chunk = manual_home_auto_confirm
                        break
                    if manual_nudge:
                        client.reset()
                        break

                    update_image_history(client, args, shm_dict, shapes, warning_state=sync_history_warning)
                    action = step_action
                    robot_action(action, shm_dict)
                    timestep += 1
                    exec_rate.sleep()

                if episode_restart_requested:
                    break

            client.close()

        if episode_restart_requested:
            action = np.zeros((action_dim,), dtype=np.float32)
            first_inference = True
            chunk_idx = 0
            if auto_confirm_next_first_chunk:
                print(
                    "[Manual Home] Rollout state has been reset. "
                    "The next chunk will be treated as a fresh start.\n"
                    "[Manual Home] Your second Enter will be reused as the first-chunk confirmation, "
                    "so execution will not wait for another prompt."
                )
            else:
                print(
                    "[Manual Home] Rollout state has been reset. "
                    "The next chunk will be treated as a fresh start.\n"
                    "[Manual Home] You will see the first-chunk confirmation again before execution."
                )

        if args.use_base and action_dim > 19:
            action[16] = 0
            action[17] = 0
            action[19] = 0

        robot_action(action, shm_dict)


__all__ = ["inference_process"]
