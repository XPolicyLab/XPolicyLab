"""
Test script to run the eval

python eval_simpler.py --test --env widowx_open_drawer
python eval_simpler.py --test --env widowx_close_drawer

# Openvla api call
python eval_simpler.py --env widowx_open_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act
python eval_simpler.py --env widowx_close_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act


# octo policy
python eval_simpler.py --env widowx_open_drawer --octo
python eval_simpler.py --env widowx_close_drawer --octo


# Example: GR00T policy

youliangtan/gr00t-n1.5-bridge-posttrain
youliangtan/gr00t-n1.5-fractal-posttrain

python scripts/inference_service.py \
    --embodiment_tag new_embodiment --denoising-steps 8 \
    --data_config examples.simpler_env.custom_data_config:FractalDataConfig \
    --model_path youliangtan/gr00t-n1.5-fractal-posttrain \
    --server --port 7799

python eval_simpler.py --env google_robot_pick_object --groot_port 7799
"""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import cv2
import numpy as np
import json
from transforms3d.euler import euler2quat
from sapien.core import Pose
from itertools import product

# for openvla api call
import requests
import json_numpy
import argparse
import cv2
import os
import numpy as np
from collections import deque
# import gymnasium as gym
import gym

try:
    import jax
except ImportError:
    print("JAX not installed.")
    print("Please install jax using `pip install jax` if you want to use Octo model.")

from transforms3d import euler as te
from transforms3d import quaternions as tq

json_numpy.patch()

print_green = lambda x: print("\033[92m {}\033[00m".format(x))

# print numpy array with 2 decimal points
np.set_printoptions(precision=2)

def view_img(obs_dict):
    """Simple image viewer for debugging"""
    for key, img in obs_dict.items():
        if isinstance(img, np.ndarray) and len(img.shape) == 3:
            cv2.imshow(f"Debug {key}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


def _parse_kv_list(kvs):
    out = {}
    for kv in kvs or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


def parse_range_tuple(t):
    if not t:
        return []
    return np.linspace(t[0], t[1], int(t[2]))


def build_reset_options(robot_init_x, robot_init_y, robot_init_quat, obj_init_x=None, obj_init_y=None, obj_episode_id=None):
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    return env_reset_options


def iter_env_resets(args):
    if len(args.robot_init_xs) == 0:
        # no variation
        yield {}
        return
    else:
        assert len(args.robot_init_xs) and len(args.robot_init_ys) and len(args.robot_init_quats)
        if args.obj_episode_range:
            # using "episode" to randomize the object position
            for x, y, q in product(args.robot_init_xs, args.robot_init_ys, args.robot_init_quats):
                for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                    yield build_reset_options(x, y, q, obj_episode_id=obj_episode_id)
            return
        else:
            # using "xy" to randomize the object position
            for x, y, q, ox, oy in product(args.robot_init_xs, args.robot_init_ys, args.robot_init_quats, args.obj_init_xs, args.obj_init_ys):
                yield build_reset_options(x, y, q, obj_init_x=ox, obj_init_y=oy)
            return


def get_maniskill2_env(robot_type, env_name, scene_name,
        additional_env_build_kwargs=None,
        control_freq=3,
        sim_freq=513,
        max_episode_steps=80,
        rgb_overlay_path=None,
    ):

    from simpler_env.utils.env.env_builder import build_maniskill2_env
    assert robot_type in ("google", "widowx"), f"Only `google` and `widowx` are supported."

    if robot_type == "google":
        control_mode = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
    elif robot_type == "widowx":
        control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    else:
        raise NotImplementedError(f"Robot {robot_type} not supported")

    kwargs = dict(
        obs_mode="rgbd",
        # Map to what maniskill2 internal APIs require.
        robot="google_robot_static" if robot_type == "google" else "widowx",
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    return env

########################################################################
class OpenVLAPolicy:
    def __init__(self, url):
        self.url = url

    def get_action(self, obs_dict, instruction):
        """
        Openvla api call to get the action.
            obs_dict : dict
            instuction : str
        """
        print("instruction", instruction)
        img = obs_dict["image_primary"]
        img = cv2.resize(img, (256, 256)) # ensure size is 256x256
        action = requests.post(
            self.url,
            json={"image": img, "instruction": instruction, "unnorm_key": "bridge_orig"},
        ).json()
        print("Action", action)
        action = np.array(action)
        return action

########################################################################

class OctoPolicy:
    def __init__(self):
        from octo.model.octo_model import OctoModel
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
        self.task = None  # created later

    def get_action(self, obs_dict, instruction):
        """
        Octo api call to get the action.
            obs_dict : dict
            instuction : str
        """
        if self.task is None:
            # assumes that each Octo model doesn't receive different tasks
            self.task = self.model.create_tasks(texts=[instruction])
            # self.task = self.agent.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        actions = self.model.sample_actions(
            jax.tree_map(lambda x: x[None], obs_dict),
            self.task,
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        actions = actions[0]  # note that actions here could be chucked
        # return actions from jax to numpy and take only the first action
        return np.asarray(actions)


########################################################################


class GR00TPolicy:
    """GR00T Policy wrapper for SimplerEnv environments.
    
    Supports WidowX and Google robots with appropriate observation and action processing.
    """

    ROBOT_CONFIGS = {
        "widowx": {
            "camera_key": "video.image_0",
            "proprio_size": 7,
            "state_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        },
        "google": {
            "camera_key": "video.image", 
            "proprio_size": 8,
            "state_keys": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        }
    }
    
    def __init__(self, host="localhost", port=5555, show_images=False, robot_type="widowx", action_horizon=1):
        # from service import ExternalRobotInferenceClient
        # from gr00t.eval.service import ExternalRobotInferenceClient
        # import from local path
        
        # NOTE: We can ensure the `service.py` is in consistent as the one in Isaac-GR00T repo. THis can be done
        # with the following code. while keeping them as different env. Else, copy the `service.py` to the local path.
        # import sys
        # import os
        # sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
        from service import ExternalRobotInferenceClient

        if robot_type not in self.ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot_type: {robot_type}. Supported: {list(self.ROBOT_CONFIGS.keys())}")
            
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.show_images = show_images
        self.robot_type = robot_type
        self.config = self.ROBOT_CONFIGS[robot_type]
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.action_horizon = action_horizon

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        obs_dict = self._process_observation(observation_dict, lang)
        action_chunk = self.policy.get_action(obs_dict)
        if self.action_horizon == 1:
            return self._convert_to_simpler_action(action_chunk, 0)
        else:
            actions = []
            for i in range(self.action_horizon):
                actions.append(self._convert_to_simpler_action(action_chunk, i))
            actions = np.stack(actions, axis=0)
            return actions
    
    def _process_observation(self, observation_dict, lang: str):
        """Convert SimplerEnv observation to GR00T format."""
        obs_dict = {}

        # Add camera image
        obs_dict[self.config["camera_key"]] = observation_dict["image_primary"]

        # Show images for debugging if enabled
        if self.show_images:
            view_img({self.config["camera_key"]: obs_dict[self.config["camera_key"]]})

        # Process proprioceptive state
        proprio = observation_dict["proprio"]
        expected_size = self.config["proprio_size"]
        assert len(proprio) == expected_size, f"Expected proprio size {expected_size}, got {len(proprio)}"
        
        # Map proprio components to state keys
        state_keys = self.config["state_keys"]
        for i, key in enumerate(state_keys):
            obs_dict[f"state.{key}"] = proprio[i:i+1].astype(np.float64)
            
        # Add padding for WidowX (required by model)
        if self.robot_type == "widowx":
            obs_dict["state.pad"] = np.array([0.0]).astype(np.float64)
        
        # Add task description
        obs_dict["annotation.human.task_description"] = lang
        
        # Add batch dimension (history=1)
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                obs_dict[key] = value[np.newaxis, ...]
            else:
                obs_dict[key] = [value]
                
        return obs_dict

    def _convert_to_simpler_action(self, action_chunk: dict[str, np.array], idx: int = 0) -> np.ndarray:
        """Convert GR00T action chunk to SimplerEnv format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] 
            for key in self.action_keys
        ]
        
        action_array = np.array(action_components, dtype=np.float32)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array

########################################################################


class WrapSimplerEnv(gym.Wrapper):
    def __init__(self, env, image_size=(256, 256)):
        super(WrapSimplerEnv, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )
        self.image_size = image_size

    def reset(self, **kwargs):
        obs, reset_info = self.env.reset(**kwargs)
        obs, additional_info = self._process_obs(obs)
        reset_info.update(additional_info)
        return obs, reset_info

    def step(self, action):
        """
        NOTE action is 7 dim
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        gripper: -1 close, 1 open
        """
        obs, reward, done, truncated, info = self.env.step(action)
        obs, additional_info = self._process_obs(obs)
        info.update(additional_info)
        return obs, reward, done, truncated, info

    def _process_obs(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs, camera_name=None)
        image_path = f"images/0.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, img)
        proprio = self._process_proprio(obs)
        return (
            {
                "image_primary": cv2.resize(img, self.image_size),
                "proprio": proprio,
            }, 
            {
                "original_image_primary": img,
            }
        )

    def _process_proprio(self, obs):
        """
        Process proprioceptive information
        """
        # TODO: should we use rxyz instead of quaternion?
        # 3 dim translation, 4 dim quaternion rotation and 1 dim gripper
        eef_pose = obs['agent']["eef_pos"]
        # joint_angles = obs['agent']['qpos'] # 8-dim vector joint angles
        return eef_pose


########################################################################

# action were post processed in the original simpler env code
#  https://github.com/simpler-env/SimplerEnv/blob/4ab7178e83e84ee06894034ec6dbf9e7aad1e882/simpler_env/policies/octo/octo_model.py#L187-L242

class GoogleSimplerActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GoogleSimplerActionWrapper, self).__init__(env)
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def step(self, action):
        action[-1] = self._postprocess_gripper(action[-1])
        obs, reward, done, trunc, info = super().step(action)
        obs["proprio"] = self._preprocess_proprio(obs["proprio"])
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        return super().reset(**kwargs)

    def _preprocess_proprio(self, proprio: np.array) -> np.array:
        # gripper, the last dimension is handled in the postprocess_gripper
        quat_xyzw = np.roll(proprio[3:7], -1)
        gripper_closedness = (1 - proprio[7])
        raw_proprio = np.concatenate(
            (
                proprio[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def _postprocess_gripper(self, current_gripper_action: float) -> float:
        current_gripper_action = (current_gripper_action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -current_gripper_action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -current_gripper_action
        # self.previous_gripper_action = current_gripper_action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action


class BridgeSimplerStateWrapper(gym.Wrapper):
    """
    NOTE(YL): this converts the prorio from the default 
    [x, y, z, qx, qy, qz, qw, gripper [0, 1]]
    is adapted from:
    https://github.com/allenzren/open-pi-zero/blob/main/src/agent/env_adapter/simpler.py
    """
    def __init__(self, env, **kwargs):
        super(BridgeSimplerStateWrapper, self).__init__(env)
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        
        # NOTE: now proprio is size 7
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["proprio"] = self._preprocess_proprio(obs)
        return obs, info

    def step(self, action):
        action[-1] = self._postprocess_gripper(action[-1])
        obs, reward, done, trunc, info = super().step(action)
        obs["proprio"] = self._preprocess_proprio(obs)
        assert len(obs["proprio"]) == 7, "propio is incorrect size"
        return obs, reward, done, trunc, info

    def _preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        # proprio = obs["agent"]["eef_pos"]
        proprio = obs["proprio"]
        assert len(proprio) == 8, "original proprio should be size 8"
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def _postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper


def run_eval_per_setting(env, env_reset_options, args) -> int:
    print(f"Evaluate with reset options: {env_reset_options}")
    success_count = 0
    for i in range(args.eval_count):
        print_green(f"Evaluate Episode {i}")

        done, truncated = False, False
        obs, info = env.reset(options=env_reset_options)

        images = []

        step_count = 0
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            # image = get_image_from_maniskill2_obs_dict(env, obs)
            image = obs["image_primary"]

            if args.output_video_dir:
                images.append(image)

            instruction = base_env.unwrapped.get_language_instruction()

            if args.test:
                # random action
                actions = env.action_space.sample()
            else:
                actions = policy.get_action(obs, instruction)

            # print(f"Step {step_count} Action: {action}")
            # show image
            for j in range(args.action_horizon):
                action = actions if args.action_horizon == 1 else actions[j]
                obs, reward, done, truncated, info = env.step(action)

                if not args.headless:
                    full_image = info["original_image_primary"]
                    cv2.imshow("Image", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        truncated = True

                if done or truncated:
                    break

            step_count += 1

        # check if the episode is successful
        if done:
            success_count += 1
            print_green(f"Episode {i} Success")
        else:
            print_green(f"Episode {i} Failed")

        # save mp4 video of the current episode
        if args.output_video_dir:
            video_name = f"{args.output_video_dir}/{args.env}_{i}.mp4"
            print(f"Save video to {video_name}")
            height, width, _ = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
            for image in images:
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

        episode_stats = info.get("episode_stats", {})
        print("Episode stats", episode_stats)
        print_green(f"Success rate: {success_count}/{i + 1}")

    print(f"env_reset_options: {env_reset_options} Success rate: {success_count}/{args.eval_count}")
    return success_count


########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Either supply with `env` or `robot_type` + `env_name` + `scene_name`.
    parser.add_argument("--env", type=str, default=None)

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--octo", action="store_true")
    parser.add_argument("--vla_url", type=str, default="http://100.76.193.18:6633/act")
    parser.add_argument("--groot_port", type=int, default=6699)
    parser.add_argument("--eval_count", type=int, default=50)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--output_video_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_horizon", type=int, default=1)
    
    # The following are for variant aggr mode.
    parser.add_argument("--robot_type", type=str, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--additional_env_build_kwargs", nargs="*", default=[],
                help='Extra key=val pairs for env build (e.g. lr_switch=True distractor_config=more)')
    parser.add_argument("--rgb_overlay_path", type=str, default=None)
    # robot and object init positions
    parser.add_argument("--robot_init_x_range", type=float, nargs=3, metavar=("MIN","MAX","STEP"),
                        help="Robot X range: min max step")
    parser.add_argument("--robot_init_y_range", type=float, nargs=3, metavar=("MIN","MAX","STEP"),
                        help="Robot Y range: min max step")
    parser.add_argument("--obj_episode_range", type=int, nargs=2, metavar=("MIN","MAX"),
                        help="Object episode range: min max")
    # 9 floats: r_min r_max r_step p_min p_max p_step y_min y_max y_step
    parser.add_argument("--robot_init_rot_rpy_range", type=float, nargs=9, metavar=("RMIN","RMAX","RSTEP","PMIN","PMAX","PSTEP","YMIN","YMAX","YSTEP"),
                        help="RPY ranges (rad): r_min r_max r_step p_min p_max p_step y_min y_max y_step")

    # center quaternion (wrt which we offset by RPY)
    parser.add_argument("--robot_init_rot_quat_center", type=float, nargs=4, default=[0,0,0,1],
                        metavar=("QX","QY","QZ","QW"), help="Center quaternion to compose with RPY")
    parser.add_argument("--obj_init_x_range", type=float, nargs=3, metavar=("MIN","MAX","STEP"),
                        help="Object X range: min max step (used if --obj_variation_mode xy)")
    parser.add_argument("--obj_init_y_range", type=float, nargs=3, metavar=("MIN","MAX","STEP"),
                        help="Object Y range: min max step (used if --obj_variation_mode xy)")
    
    args = parser.parse_args()

    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3] if args.robot_init_rot_rpy_range else None):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6] if args.robot_init_rot_rpy_range else None):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:] if args.robot_init_rot_rpy_range else None):
                args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    # env args: object position
    args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
    args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)

    robot_type = None
    if args.env:
        # run visual matching evaluation
        assert args.robot_type is None and args.env_name is None and args.scene_name is None, "Either supply with `env` or `robot_type` + `env_name` + `scene_name`. But not both."
        base_env = simpler_env.make(args.env)
        robot_type = "google" if "google" in args.env else "widowx"
    else:
        assert args.robot_type is not None and args.env_name is not None and args.scene_name is not None, "Either supply with `env` or `robot_type` + `env_name` + `scene_name`. But not both."
        build_kwargs = _parse_kv_list(args.additional_env_build_kwargs)
        robot_type = args.robot_type
        assert robot_type in ["google", "widowx"], f"Only `google` and `widowx` are supported."
        base_env = get_maniskill2_env(robot_type, args.env_name, args.scene_name, build_kwargs, max_episode_steps=args.episode_length, rgb_overlay_path=args.rgb_overlay_path)

    base_env._max_episode_steps = args.episode_length # override the max episode length

    instruction = base_env.unwrapped.get_language_instruction()

    env = WrapSimplerEnv(base_env)

    if robot_type == "widowx":
        print("Wrap Simpler with bridge state wrapper for proprio and action convention")
        env = BridgeSimplerStateWrapper(env)
    elif robot_type == "google":
        print("Wrap Simpler with google action wrapper for sticky gripper")
        env.image_size = (320, 256) # wrap the image size to "320, 256"
        env = GoogleSimplerActionWrapper(env)

    print("Instruction", instruction)

    if not args.test:
        if args.octo:
            policy = OctoPolicy()
            from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper

            env = HistoryWrapper(env, horizon=2)  # Expects action_horizon to be 2 for octo
            env = TemporalEnsembleWrapper(env, 4)
        elif args.groot_port:
            policy = GR00TPolicy(port=args.groot_port, robot_type=robot_type, action_horizon=args.action_horizon)
        else:
            policy = OpenVLAPolicy(args.vla_url)

    success_count = 0

    aggr_eval_count = 0
    for reset_options in iter_env_resets(args):
        success_count += run_eval_per_setting(env, reset_options, args)
        aggr_eval_count += args.eval_count

    print(f"Final Success rate: {success_count}/{aggr_eval_count}")
