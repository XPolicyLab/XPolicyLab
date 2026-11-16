"""Own ARX X5 geometry extracted from the development controller.

Only public robot/camera calibration and observed robot state are used.
"""

import ast
import hashlib
import io
from pathlib import Path
import warnings

from ikpy.chain import Chain
from ikpy.link import URDFLink
import numpy as np
from PIL import ImageFile
from scipy.spatial.transform import Rotation
import yaml

from orimapi.state import RGBImage

RESOURCE_KEYS = (
    "robot_urdf",
    "robot_config",
    "arm_config",
    "camera_config",
    "camera_template",
    "table_config",
)


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def pose_matrix(pose):
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite world pose [x,y,z,w,x,y,z]")
    if abs(np.linalg.norm(value[3:]) - 1) > 0.01:
        raise ValueError("Invalid observed orientation")
    return transform(value[:3], Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix())


def rgb(image):
    if isinstance(image, RGBImage):
        return (
            np.asarray(memoryview(image.pixels), dtype=np.uint8)
            .reshape(image.height, image.width, 3)
            .copy()
        )
    parser = ImageFile.Parser()
    parser.feed(image)
    with parser.close() as value:
        return np.asarray(value.convert("RGB"), dtype=np.uint8)


class Camera:
    def __init__(self, pose, template):
        self.pose = np.asarray(pose)
        self.width, self.height = template["resolution"]
        focal = self.width * template["focal_length"] / template["horizontal_aperture"]
        self.k = np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        )
        # USD cameras look along -Z with +Y up; pixels use OpenCV axes.
        self.optical = self.pose @ np.diag([1.0, -1.0, -1.0, 1.0])
        self.view = np.linalg.inv(self.optical)
        self.projection = self.k @ self.view[:3]

    def project(self, point):
        xyz = self.view @ np.r_[point, 1.0]
        if xyz[2] <= 0:
            raise ValueError("Point is behind camera")
        pixel = self.k @ xyz[:3]
        return pixel[:2] / pixel[2]

    def on_plane(self, pixel, height):
        ray = self.optical[:3, :3] @ np.linalg.solve(self.k, np.r_[pixel, 1.0])
        if abs(ray[2]) < 1e-6:
            raise ValueError("Camera ray is parallel to plane")
        distance = (height - self.optical[2, 3]) / ray[2]
        if distance <= 0:
            raise ValueError("Plane is behind camera")
        return self.optical[:3, 3] + distance * ray


class RobotGeometry:
    def __init__(self, parameters):
        self.resources = {
            key: Path(parameters[key]).expanduser().resolve(strict=True)
            for key in RESOURCE_KEYS
        }
        self.hashes = {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in self.resources.items()
        }
        if parameters.get("robot_collision_config"):
            path = Path(parameters["robot_collision_config"]).expanduser().resolve(strict=True)
            self.resources["robot_collision_config"] = path
            self.hashes["robot_collision_config"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.robot_config = yaml.safe_load(self.resources["robot_config"].read_text())
        robots = yaml.safe_load(self.resources["arm_config"].read_text())["robots"]
        self.bases = {
            side: pose_matrix(row["default_root_pos"] + row["default_root_rot"])
            for side, row in zip(("left", "right"), robots, strict=True)
        }
        self.camera_config = yaml.safe_load(self.resources["camera_config"].read_text())
        self.templates = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(self.resources["camera_template"].read_text()).body
            if isinstance(node, ast.Assign)
        }
        table = yaml.safe_load(self.resources["table_config"].read_text())["Table"]
        self.table = float(table["default_pos"][2] + table["scale"][2] / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            full = Chain.from_urdf_file(
                str(self.resources["robot_urdf"]), symbolic=False
            )
        if [link.name for link in full.links[1:7]] != self.robot_config[
            "arm_joints_name"
        ]:
            raise ValueError("Unexpected robot joint chain")
        self.tip_offset = float(self.robot_config["gripper_bias"])
        tip = URDFLink(
            name="grasp_point",
            origin_translation=[self.tip_offset, 0, 0],
            origin_orientation=[0, 0, 0],
            joint_type="fixed",
            use_symbolic_matrix=False,
        )
        self.chain = Chain(
            full.links[:7] + [tip], active_links_mask=[False] + [True] * 6 + [False]
        )
        for link in self.chain.links[1:7]:
            link.bounds = (max(link.bounds[0], -np.pi), min(link.bounds[1], np.pi))
        camera_links = full.links[7:]
        if [link.name for link in camera_links] != [
            "hand_to_camera_mount",
            "camera_joint",
        ]:
            raise ValueError("Unexpected wrist camera chain")
        self.wrist_mount = np.eye(4)
        for link in camera_links:
            self.wrist_mount = self.wrist_mount @ link.get_link_frame_matrix(0)
        wrist = self.robot_config["camera"][0]
        self.wrist_mount = self.wrist_mount @ transform(
            wrist["pos"],
            Rotation.from_euler("XYZ", wrist["ori"], degrees=True).as_matrix(),
        )

    def fk(self, side, joints, *, full=False):
        local = self.chain.forward_kinematics(np.r_[0, joints, 0], full_kinematics=full)
        return (
            [self.bases[side] @ value for value in local]
            if full
            else self.bases[side] @ local
        )

    def cameras(self, state):
        head = self.camera_config["cam_head"]["camera"]
        result = {
            "cam_high": Camera(
                transform(
                    head["pos"],
                    Rotation.from_euler("XYZ", head["ori"], degrees=True).as_matrix(),
                ),
                self.templates["GEMINI_345LG"],
            )
        }
        for side in ("left", "right"):
            ee = pose_matrix(state[f"{side}_ee_pose"])
            result[f"cam_{side}_wrist"] = Camera(
                ee @ self.wrist_mount, self.templates["D435"]
            )
        return result

    def solve(self, side, point, rotation, start):
        target = np.linalg.inv(self.bases[side]) @ transform(point, rotation)
        choices = []
        seeds = [
            start,
            [0, 1, 1, 0, 0, 0],
            [0, 1.7, 1.3, -0.5, 0.5, 0],
            [0, -1, -1, 0, 0, 0],
        ]
        for seed in seeds:
            solution = self.chain.inverse_kinematics_frame(
                target, initial_position=np.r_[0, seed, 0], orientation_mode="all"
            )[1:7]
            actual = self.fk(side, solution)
            position_error = np.linalg.norm(actual[:3, 3] - point)
            angle_error = Rotation.from_matrix(actual[:3, :3].T @ rotation).magnitude()
            if position_error < 0.003 and angle_error < 0.04:
                choices.append(solution)
                if np.linalg.norm(solution - start) < 0.5:
                    break
        if not choices:
            raise ValueError("No accurate IK solution")
        return min(choices, key=lambda q: np.linalg.norm(q - start))
