from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv

print_green = lambda x: print("\033[92m {}\033[00m".format(x))

####################################################################################

BACKGROUND_IMAGE_PATH = "real_inpainting/bridge_small_drawer.png"

# Joint Configs of the drawer
# NOTE: tune this value 0 - 0.12, limit defined in urdf
CLOSE_DRAWER_JOINT_DIST = 0.02
OPEN_DRAWER_JOINT_DIST = 0.12
TARGET_DRAWER_TOLERANCE = 0.02

# Location of the drawer
# NOTE: x -> -ve front of the robot, y -> left of the robot, z -> up
DRAWER_XYZ = [-0.26, -0.17, 0.85]
DRAWER_RPY = [0, 0, 1.35]

####################################################################################

class OpenSmallDrawerInSceneEnv(CustomSceneEnv):
    def __init__(
        self,
        light_mode: Optional[str] = None,
        camera_mode: Optional[str] = None,
        station_name: float = "small_drawer",
        cabinet_joint_friction: float = 0.05,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        self.station_name = station_name
        self.cabinet_joint_friction = cabinet_joint_friction
        self.episode_stats = None

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / BACKGROUND_IMAGE_PATH
        )  # dummy path; to be replaced later
        # ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "small_drawer"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        print("\n\nret", ret)

        return ret

    ####################################################################################
    # We Randomize the initial pose of the robot, drawer, and background in the scene

    def _initialize_agent(self):
        # NOTE: init qpos for widowx defined in base_env.py
        init_qpos = np.array([-0.01840777,  0.0398835,   0.22242722,  -0.00460194,  1.36524296,  0.00153398, 0.037, 0.037])
        # add random offset to the initial qpos xyz
        init_qpos[:3] += np.random.uniform(-0.05, 0.05, size=3)
        # random open or close gripper
        init_qpos[-1] = np.random.choice([0.0, 1.0])

        self.robot_init_options.setdefault("qpos", init_qpos)
        super()._initialize_agent()

    def _initialize_articulations(self):
        # NOTE: Randomize the cabinet pose
        _xyz = DRAWER_XYZ.copy() + np.random.uniform(-0.01, 0.01, size=3)
        _rpy = DRAWER_RPY.copy() + np.random.uniform(-0.01, 0.01, size=3)
        self.art_obj.set_pose(sapien.Pose(_xyz, euler2quat(*_rpy)))
        return super()._initialize_articulations()

    def _additional_prepackaged_config_reset(self, options):
        background_path = str(ASSET_DIR / BACKGROUND_IMAGE_PATH)
        self.rgb_overlay_img = (
            cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
            / 255
        )
        # NOTE: we randomize the brightness of the overlay image
        self.rgb_overlay_img = np.clip(
            self.rgb_overlay_img + np.random.uniform(-0.1, 0.1), 0, 1
        )

        # NOTE: not used for now
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                # "recolor_tabletop_visual_matching_1",
                # "recolor_tabletop_visual_matching_2",
                # "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            print_green(f"Switching to URDF version: {new_urdf_version}")
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    ####################################################################################

    def _setup_lighting(self):
        if self.light_mode != "simple":
            return self._setup_lighting_legacy()

        self._scene.set_ambient_light([1.0, 1.0, 1.0])
        angle = 75
        self._scene.add_directional_light(
            [-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [1.0, 1.0, 1.0]
        )

    def _setup_lighting_legacy(self):
        # self.enable_shadow = True
        # super()._setup_lighting()

        direction = [-0.2, 0, -1]
        if self.light_mode == "vertical":
            direction = [-0.1, 0, -1]

        color = [1, 1, 1]
        if self.light_mode == "darker":
            color = [0.5, 0.5, 0.5]
        elif self.light_mode == "brighter":
            color = [2, 2, 2]

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            direction, color, shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([-1, 1, -0.05], [0.5] * 3)
        self._scene.add_directional_light([-1, -1, -0.05], [0.5] * 3)

    def _load_actors(self):
        self._load_arena_helper(add_collision=False)

    def _load_articulations(self):
        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        # TODO: This pose can be tuned for different rendering approachs.
        # convert rpy to quaternion
        quat = euler2quat(*DRAWER_RPY)
        self.art_obj.set_pose(sapien.Pose(DRAWER_XYZ, quat))
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"small_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"small_drawer_joint")

    def reset(self, seed=None, options=None, drawer_joint_distance=CLOSE_DRAWER_JOINT_DIST):
        if options is None:
            options = dict()
        options = options.copy()

        reconfigure = options.get("reconfigure", False)
        self.set_episode_rng(seed)

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options) # articulations are loaded here
        self.joint_idx = self.joint_names.index(f"small_drawer_joint")

        # setup cabinet qpos
        self.art_obj.set_qpos([drawer_joint_distance] * self.art_obj.dof) # ensure that the drawer is closed

        obs = self.get_obs()

        info.update(
            {
                "drawer_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.drawer_obj.pose,
                "cabinet_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.art_obj.pose,
                "station_name": self.station_name,
                "light_mode": self.light_mode,
            }
        )
        return obs, info


    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(qpos=0.0)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(
            success=qpos >= OPEN_DRAWER_JOINT_DIST - TARGET_DRAWER_TOLERANCE,
            qpos=qpos,
            episode_stats=self.episode_stats
        )

    def get_language_instruction(self, **kwargs):
        return "open the drawer"


@register_env("OpenSmallDrawerCustomInScene-v0", max_episode_steps=120)
class OpenSmallDrawerCustomInSceneEnv(OpenSmallDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass


@register_env("CloseSmallDrawerCustomInScene-v0", max_episode_steps=120)
class CloseSmallDrawerInSceneEnv(OpenSmallDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            options["obj_init_options"]["cabinet_init_qpos"] = OPEN_DRAWER_JOINT_DIST
        return super().reset(seed=seed, options=options, drawer_joint_distance=OPEN_DRAWER_JOINT_DIST)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        eval = dict(
            success=qpos <= CLOSE_DRAWER_JOINT_DIST + TARGET_DRAWER_TOLERANCE,
            qpos=qpos,
            episode_stats=self.episode_stats
        )
        # print("eval: ", eval)
        return eval

    def get_language_instruction(self):
        return f"close the drawer"
