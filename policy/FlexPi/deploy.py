import contextlib
import json
import os
import re
import sys
import time

import numpy as np


def eval_one_episode(TASK_ENV, model_client):
    _install_ik_probe(TASK_ENV)
    _IK["stats"].clear()

    model_client.call(func_name="reset") # reset policy

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        _check_action_label(TASK_ENV, actions[0])

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

    ik = _ik_episode(0)
    if ik:
        print(f"[ik] {_ik_line(ik)}", flush=True)

def eval_one_episode_batch(TASK_ENV, model_client):
    _install_ik_probe(TASK_ENV)
    timer = _BatchTimer(TASK_ENV)

    model_client.call(func_name="reset")

    while not timer.episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        with timer.span("obs"):
            obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
            model_client.call(func_name="update_obs_batch", obs=obs_list)
        with timer.span("policy"):
            actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action according to observation chunk
        _check_action_label(TASK_ENV, actions[0][0])
        timer.policy_calls += 1
        timer.env_infers += len(env_idx_list)

        chunk_size = len(actions[0]) # Get the chunk size
        for action_idx in range(chunk_size): # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions] # Get the current action list
            with timer.span("sim"):
                TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

            if timer.episode_end() or action_idx + 1 == chunk_size: # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running] # Get the active batch index

            actions = [actions[i] for i in active_batch_idx] # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx] # Get the active environment index list
            with timer.span("obs"):
                model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)) # Update the observation

    timer.finish()


# --- Action-type label ---------------------------------------------------------
# The simulator executes whatever the policy server sends: it tells joint from ee
# by each action dict's keys. The client's --action-type only names the result
# directory (<seed>_ckpt_name=...,action_type=...). So a server started with one
# and a client with the other would file results under the wrong label; the first
# batch of each client process checks that the two agree. It exits instead of
# raising, because main.py turns an exception in the eval loop into a silently
# skipped batch; rc 2 is not one of eval_policy.sh's restart codes.
_LABEL = {"checked": False}


def _check_action_label(task_env, action):
    if _LABEL["checked"]:
        return
    _LABEL["checked"] = True
    sent = "ee" if any(str(key).endswith("ee_pose") for key in action) else "joint"
    label = re.search(r"action_type=([^,/\s]+)", str(getattr(task_env, "additional_info", "") or ""))
    if label is None or label.group(1) == sent:
        return
    print(
        f"[FlexPi] FATAL: the policy server sends {sent} actions but this client run is labelled "
        f"action_type={label.group(1)}, so its results would be filed under the wrong action type. "
        "Start the server and the client with the same --action-type.",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(2)


# --- IK accounting (action_type: ee) ----------------------------------------------
# RoboDojo runs an ee action as one cuRobo IK solve per arm per 25 Hz step
# (eval_env.take_action_batch -> robot_manager.solve_ik). A failed solve is not
# reported anywhere: the arm key is dropped, ControlManager refills it from
# prev_control, and the arm just holds its last command. This probe wraps
# solve_ik on the env's robot_manager and keeps, per (env, arm):
#   calls, fails and the longest run of consecutive fails;
#   max |IK joints - current joints|, and how many solves exceed _IK_JUMP_RAD.
#   One step moves a joint at most ~0.2 rad (5 rad/s actuator limit), so a jump
#   well past that is IK switching branch or wrapping a joint by 2 pi (the X5
#   URDF allows +-10 rad on joints 1-5).
# Joint actions never call solve_ik, so under joint nothing is recorded. Like the
# timing below it never fails the eval: any error turns the probe off.
_IK = {"stats": {}, "off": False}
_IK_JUMP_RAD = 0.5


def _install_ik_probe(task_env):
    robot_manager = getattr(task_env, "robot_manager", None)
    if _IK["off"] or robot_manager is None or getattr(robot_manager, "_flexpi_ik_probe", False):
        return
    try:
        solve = robot_manager.solve_ik

        def solve_ik(target_pose, env_idx, robot, *args, **kwargs):
            result = solve(target_pose, env_idx, robot, *args, **kwargs)
            if not _IK["off"]:
                try:
                    _record_ik(robot_manager, result, env_idx, robot)
                except Exception as exc:
                    _IK["off"] = True
                    print(f"[ik] probe off ({exc!r}); IK itself is unaffected", flush=True)
            return result

        robot_manager.solve_ik = solve_ik
        robot_manager._flexpi_ik_probe = True
    except Exception as exc:
        _IK["off"] = True
        print(f"[ik] could not install the probe: {exc!r}", flush=True)


def _record_ik(robot_manager, result, env_idx, robot):
    stats = _IK["stats"].setdefault(
        (int(env_idx), str(robot.arm_name)),
        {"calls": 0, "fails": 0, "fail_run": 0, "max_fail_run": 0, "max_dq": 0.0, "jumps": 0},
    )
    stats["calls"] += 1
    if result.get("status") != "Success":
        stats["fails"] += 1
        stats["fail_run"] += 1
        stats["max_fail_run"] = max(stats["max_fail_run"], stats["fail_run"])
        return
    stats["fail_run"] = 0
    q_now = np.asarray(robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx], dtype=np.float64)
    q_ik = np.asarray(result["joint_value"], dtype=np.float64)
    dq = float(np.abs(q_ik.reshape(-1) - q_now.reshape(-1)).max())
    stats["max_dq"] = max(stats["max_dq"], dq)
    stats["jumps"] += int(dq > _IK_JUMP_RAD)


def _ik_episode(env_idx):
    """``{arm: stats}`` for one env since the last clear, or None if it never ran IK."""
    return {
        arm: {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items() if k != "fail_run"}
        for (idx, arm), stats in sorted(_IK["stats"].items())
        if idx == int(env_idx)
    } or None


def _ik_line(ik):
    return "  ".join(
        f"{arm} fails {s['fails']}/{s['calls']} (longest run {s['max_fail_run']}), "
        f"max|dq| {s['max_dq']:.3f} rad, {s['jumps']} over {_IK_JUMP_RAD}"
        for arm, s in ik.items()
    )


# --- Wall-clock timing ---------------------------------------------------------
# One JSON line per batch in <save_dir>/_timing.jsonl (next to _result.json), plus a
# summary in the task log. The rollout is split into policy (get_action_batch round
# trips: serial inference over the running envs + transport), obs (get_obs_batch +
# update_obs_batch) and sim (take_action_batch + is_episode_end). An episode's wall_s
# runs from its batch's start to the step it ended: envs in a batch step together,
# so that includes the cost of stepping its batch-mates. Timing never fails the eval.
_TIMING = {"batch": 0, "last_end": None}


def _process_age_s():
    """Seconds since this process started (includes the Isaac Sim launch)."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


class _BatchTimer:
    def __init__(self, task_env):
        self.env = task_env
        self.t0 = time.perf_counter()
        # First batch of this process: the time to get here is Isaac launch + scene
        # build + first reset. Later batches: previous batch's video save + reset.
        self.startup_s = _process_age_s() if _TIMING["last_end"] is None else None
        self.gap_before_s = None if _TIMING["last_end"] is None else self.t0 - _TIMING["last_end"]
        self.spent = {"policy": 0.0, "obs": 0.0, "sim": 0.0}
        self.policy_calls = 0
        self.env_infers = 0
        self.end_s = {}
        self.running = set(task_env.get_running_env_idx_list())
        _IK["stats"].clear()  # IK counts are per batch, like the episodes rows

    @contextlib.contextmanager
    def span(self, key):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.spent[key] += time.perf_counter() - t

    def episode_end(self):
        with self.span("sim"):
            done = self.env.is_episode_end()
        now = set(self.env.get_running_env_idx_list())
        for env_idx in self.running - now:
            self.end_s[env_idx] = time.perf_counter() - self.t0
        self.running = now
        return done

    def finish(self):
        t_end = time.perf_counter()
        _TIMING["last_end"] = t_end
        batch = _TIMING["batch"]
        _TIMING["batch"] += 1
        try:
            env = self.env
            seed_map = getattr(env, "current_env_seed_map", {})
            episodes = [
                {
                    "env_idx": int(i),
                    "layout_id": int(seed_map[i]),
                    "steps": int(env.take_action_cnt[i]),
                    "wall_s": round(self.end_s.get(i, t_end - self.t0), 3),
                    **({"ik": _ik_episode(i)} if _ik_episode(i) else {}),
                }
                for i in sorted(seed_map)
            ]
            rec = {
                "batch": batch,
                "startup_s": None if self.startup_s is None else round(self.startup_s, 3),
                "gap_before_s": None if self.gap_before_s is None else round(self.gap_before_s, 3),
                "rollout_s": round(t_end - self.t0, 3),
                **{f"{k}_s": round(v, 3) for k, v in self.spent.items()},
                "policy_calls": self.policy_calls,
                "env_infers": self.env_infers,
                "episodes": episodes,
            }
            # The debug env client has no result dir; print the summary only.
            save_dir = getattr(env, "save_dir", None)
            if save_dir:
                with open(os.path.join(save_dir, "_timing.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            ms_per_infer = 1000.0 * rec["policy_s"] / max(self.env_infers, 1)
            head = f"startup {rec['startup_s']}s" if batch == 0 else f"gap {rec['gap_before_s']}s"
            print(
                f"[timing] batch {batch}  envs {len(episodes)}  rollout {rec['rollout_s']:.1f}s = "
                f"sim {rec['sim_s']:.1f} + obs {rec['obs_s']:.1f} + policy {rec['policy_s']:.1f} "
                f"({self.env_infers} infers, {ms_per_infer:.0f} ms/infer)  {head}",
                flush=True,
            )
            for ep in episodes:
                print(
                    f"[timing]   layout {ep['layout_id']}  steps {ep['steps']}  {ep['wall_s']:.1f}s",
                    flush=True,
                )
                if "ik" in ep:
                    print(f"[ik]   layout {ep['layout_id']}  {_ik_line(ep['ik'])}", flush=True)
        except Exception as exc:
            print(f"[timing] could not record batch {batch}: {exc!r}", flush=True)