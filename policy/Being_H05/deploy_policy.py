"""
Remote RoboTwin deploy_policy for Being-H05 using Being-H inference server.
"""

import os
import sys
import numpy as np

BEINGH_ROOT = os.path.join(os.path.dirname(__file__), "Being-H")
if BEINGH_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(BEINGH_ROOT))


def summarize_obs_array(name, arr):
    arr = np.asarray(arr)
    return (
        f"{name}:shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={float(arr.min()):.3f} max={float(arr.max()):.3f} mean={float(arr.mean()):.3f} "
        f"all_zero={bool(np.all(arr == 0))}"
    )


def encode_obs(observation):
    obs_dict = observation["observation"]
    head_rgb = obs_dict["head_camera"]["rgb"]
    right_rgb = obs_dict["right_camera"]["rgb"]
    left_rgb = obs_dict["left_camera"]["rgb"]

    qpos = np.array(observation["joint_action"]["vector"], dtype=np.float32)

    return {
        "video.head_view": head_rgb,
        "video.right_wrist_view": right_rgb,
        "video.left_wrist_view": left_rgb,
        "state.left_arm_joint_position": qpos[0:6],
        "state.left_gripper_position": qpos[6:7],
        "state.right_arm_joint_position": qpos[7:13],
        "state.right_gripper_position": qpos[13:14],
    }


class RemoteBeingHPolicy:
    def __init__(self, host, port, api_token=None, enable_rtc=False):
        from BeingH.inference.beingh_service import BeingHInferenceClient

        self.client = BeingHInferenceClient(host=host, port=port, api_token=api_token or None)
        self._instruction = None
        self._obs_cache = None
        self.enable_rtc = enable_rtc
        self._prev_chunk = None
        self._inference_delay = 0

        if not self.client.ping():
            raise ConnectionError(f"Failed to reach Being-H server at {host}:{port}")

    def get_action(self, observations):
        if self.enable_rtc and self._prev_chunk is not None:
            observations = dict(observations)
            observations["prev_chunk"] = self._prev_chunk
            observations["inference_delay"] = self._inference_delay

        result = self.client.get_action(observations)
        if self.enable_rtc and "action_unified" in result:
            self._prev_chunk = np.array(result["action_unified"], dtype=np.float32)
        return result

    def reset(self):
        self._instruction = None
        self._obs_cache = None
        self._prev_chunk = None
        self._inference_delay = 0


def get_model(usr_args):
    host = usr_args.get("server_host", "127.0.0.1")
    port = int(usr_args.get("server_port", 5557))
    api_token = usr_args.get("api_token", "")
    enable_rtc = bool(usr_args.get("enable_rtc", False))
    return RemoteBeingHPolicy(host=host, port=port, api_token=api_token, enable_rtc=enable_rtc)


def eval(TASK_ENV, model, observation):
    if model._instruction is None:
        model._instruction = TASK_ENV.get_instruction()

    beingh_obs = {"language.instruction": model._instruction}
    beingh_obs.update(encode_obs(observation))

    print(f"[OBSDEBUG] instruction={model._instruction}")
    print('[OBSDEBUG] ' + summarize_obs_array('head_view', beingh_obs['video.head_view']))
    print('[OBSDEBUG] ' + summarize_obs_array('right_wrist_view', beingh_obs['video.right_wrist_view']))
    print('[OBSDEBUG] ' + summarize_obs_array('left_wrist_view', beingh_obs['video.left_wrist_view']))
    current_state = np.concatenate([
        beingh_obs['state.left_arm_joint_position'],
        beingh_obs['state.left_gripper_position'],
        beingh_obs['state.right_arm_joint_position'],
        beingh_obs['state.right_gripper_position'],
    ], axis=-1)
    print(
        f"[OBSDEBUG] state_shape={tuple(current_state.shape)} state_min={float(current_state.min()):.6f} "
        f"state_max={float(current_state.max()):.6f} state_mean={float(current_state.mean()):.6f} "
        f"state_all_zero={bool(np.all(current_state == 0))} state={current_state.tolist()}"
    )

    result = model.get_action(beingh_obs)

    current_qpos = np.concatenate(
        [
            beingh_obs["state.left_arm_joint_position"],
            beingh_obs["state.left_gripper_position"],
            beingh_obs["state.right_arm_joint_position"],
            beingh_obs["state.right_gripper_position"],
        ],
        axis=-1,
    )

    left_arm = np.array(result["action.left_arm_joint_position"])
    left_grip = np.array(result["action.left_gripper_position"])
    right_arm = np.array(result["action.right_arm_joint_position"])
    right_grip = np.array(result["action.right_gripper_position"])
    actions = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=-1)

    print(
        f"[DEBUG] obs_qpos_range=({current_qpos.min():.4f}, {current_qpos.max():.4f}) "
        f"first_action_range=({actions[0].min():.4f}, {actions[0].max():.4f}) "
        f"chunk_len={len(actions)}"
    )
    if len(actions) > 1:
        step_delta = actions[1:] - actions[:-1]
        print(
            f"[DEBUG] inter_step_delta_abs_max={np.abs(step_delta).max():.4f} "
            f"inter_step_delta_abs_mean={np.abs(step_delta).mean():.4f}"
        )

    for i in range(len(actions)):
        if TASK_ENV.eval_success:
            break
        TASK_ENV.take_action(actions[i], action_type="qpos")
        observation = TASK_ENV.get_obs()


def reset_model(model):
    model.reset()
