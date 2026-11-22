import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
import torch
from abc import ABC, abstractmethod


def _to_numpy_float32(x):
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def check_points_proj(points):  # Fomulate any form of D-dim points input into [N, 1, D]
    x = np.array(points)
    if len(x.shape) == 1:
        return x[None, None, :]
    elif len(x.shape) == 2:
        return x[:, None, :]
    return x


def intrinsics_roi_map(K_src, scale=2.25, crop_x=240.0, crop_y=0.0, crop_mode="pre"):
    K = K_src.astype(np.float64).copy()
    s = float(scale)
    cx, cy = float(K[0, 2]), float(K[1, 2])

    K[0, 0] /= s
    K[1, 1] /= s

    if crop_mode == "pre":
        K[0, 2] = (cx - float(crop_x)) / s
        K[1, 2] = (cy - float(crop_y)) / s
    elif crop_mode == "post":
        K[0, 2] = cx / s - float(crop_x)
        K[1, 2] = cy / s - float(crop_y)
    else:
        raise ValueError("crop_mode must be 'pre' or 'post'")
    return K.astype(np.float64)


def n32(x):
    return np.array(x, dtype=np.float32)


def n64(x):
    return np.array(x, dtype=np.float64)


def resize_camera_intrinsics(K_src, src_wh, dst_wh, resize_mode="crop"):
    if resize_mode == "crop":
        scale = min(src_wh[0] / dst_wh[0], src_wh[1] / dst_wh[1])
        crop_x = (src_wh[0] - dst_wh[0] * scale) / 2.0
        crop_y = src_wh[1] - dst_wh[1] * scale
        r = intrinsics_roi_map(
            K_src, scale=scale, crop_x=crop_x, crop_y=crop_y, crop_mode="pre"
        )
    elif resize_mode == "resize":
        scale_w, scale_h = src_wh[0] / dst_wh[0], src_wh[1] / dst_wh[1]

        K = K_src.astype(np.float64).copy()
        K[0, 0] /= scale_w
        K[0, 2] /= scale_w

        K[1, 1] /= scale_h
        K[1, 2] /= scale_h

        r = K
    else:
        raise NotImplementedError
    return r


def npzero3():
    return np.zeros(3, dtype=np.float64)


def nvec(x):
    return x / np.sqrt(np.sum(x**2))


def apply_T(T, p3):
    p4 = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)
    out = T @ p4
    return out[:3]


def read_cameras_for_bimanual_3views(
    file_path, target_wh_or_whs=(224, 176), resize_mode="crop"
):  # choose head_coord as unified world coordinates
    # file_path points to the caller's camera calibration JSON.
    with open(file_path, "r") as f:
        result = json.loads(f.read())
    if isinstance(result["calibration"]["intrinsics"], str):
        result["calibration"]["intrinsics"] = json.loads(
            result["calibration"]["intrinsics"]
        )

    if isinstance(result["calibration"]["extrinsics"], str):
        result["calibration"]["extrinsics"] = json.loads(
            result["calibration"]["extrinsics"]
        )

    intrinsics = result["calibration"]["intrinsics"]
    extrinsics = result["calibration"]["extrinsics"]["transform"]
    if isinstance(target_wh_or_whs, dict):
        target_whs = target_wh_or_whs
    else:
        target_whs = {
            "face_view": target_wh_or_whs,
            "left_wrist_view": target_wh_or_whs,
            "right_wrist_view": target_wh_or_whs,
        }
    if isinstance(resize_mode, dict):
        resize_mode = resize_mode
    else:
        resize_mode = {
            "face_view": resize_mode,
            "left_wrist_view": resize_mode,
            "right_wrist_view": resize_mode,
        }
    head_camera = Bimanual3ViewsCamera(
        np.eye(4, dtype=np.float64),
        np.eye(4, dtype=np.float64),
        resize_camera_intrinsics(
            n64(intrinsics["head"]["camera_matrix"]),
            (intrinsics["head"]["image_width"], intrinsics["head"]["image_height"]),
            target_whs["face_view"],
            resize_mode["face_view"],
        ),
        intrinsics["head"]["lens_model"],
        n64(intrinsics["head"]["dist_coeffs"]),
        image_width=target_whs["face_view"][0],
        image_height=target_whs["face_view"][1],
        params_name={
            "camera_name": "head",
            "extrinsics": {
                "pre_transform_partial": {"name": "None", "valid": False},
                "extrinsics_partial": {"name": "None", "valid": False},
            },
            "intrinsics": {
                "intrinsics_partial": {"name": "camera_matrix", "valid": True}
            },
        },
    )

    def proj_follow_left_pos_n_rot_to_head_coord(
        t3,
        r3,
        p3,
        M1=n64(extrinsics["M_gripper_to_camL"]),
        M2=np.linalg.inv(n64(extrinsics["M_left_to_head"])),
    ):
        p3 = Rotation.from_euler("xyz", r3).as_matrix() @ p3
        return apply_T(M2, apply_T(M1, p3))

    def proj_follow_right_pos_n_rot_to_head_coord(
        t3,
        r3,
        p3,
        M1=n64(extrinsics["M_gripper_to_camR"]),
        M2=np.linalg.inv(n64(extrinsics["M_right_to_head"])),
    ):
        p3 = Rotation.from_euler("xyz", r3).as_matrix() @ p3
        return apply_T(M2, apply_T(M1, p3))

    left_camera = Bimanual3ViewsCamera(
        np.linalg.inv(n64(extrinsics["M_gripper_to_camL"]))
        @ n64(extrinsics["M_left_to_head"]),
        n64(extrinsics["M_gripper_to_camL"]),
        resize_camera_intrinsics(
            n64(intrinsics["left"]["camera_matrix"]),
            (intrinsics["left"]["image_width"], intrinsics["left"]["image_height"]),
            target_whs["left_wrist_view"],
            resize_mode["left_wrist_view"],
        ),
        intrinsics["left"]["lens_model"],
        n64(intrinsics["left"]["dist_coeffs"]),
        image_width=target_whs["left_wrist_view"][0],
        image_height=target_whs["left_wrist_view"][1],
        params_name={
            "camera_name": "left",
            "extrinsics": {
                "pre_transform_partial": {"name": "M_left_to_head", "valid": True},
                "extrinsics_partial": {"name": "M_gripper_to_camL", "valid": True},
            },
            "intrinsics": {
                "intrinsics_partial": {"name": "camera_matrix", "valid": True}
            },
        },
    )
    right_camera = Bimanual3ViewsCamera(
        np.linalg.inv(n64(extrinsics["M_gripper_to_camR"]))
        @ n64(extrinsics["M_right_to_head"]),
        n64(extrinsics["M_gripper_to_camR"]),
        resize_camera_intrinsics(
            n64(intrinsics["right"]["camera_matrix"]),
            (intrinsics["right"]["image_width"], intrinsics["right"]["image_height"]),
            target_whs["right_wrist_view"],
            resize_mode["right_wrist_view"],
        ),
        intrinsics["right"]["lens_model"],
        n64(intrinsics["right"]["dist_coeffs"]),
        image_width=target_whs["right_wrist_view"][0],
        image_height=target_whs["right_wrist_view"][1],
        params_name={
            "camera_name": "right",
            "extrinsics": {
                "pre_transform_partial": {"name": "M_right_to_head", "valid": True},
                "extrinsics_partial": {"name": "M_gripper_to_camR", "valid": True},
            },
            "intrinsics": {
                "intrinsics_partial": {"name": "camera_matrix", "valid": True}
            },
        },
    )
    return {
        "camera_face_view": head_camera,
        "camera_left_wrist_view": left_camera,
        "camera_right_wrist_view": right_camera,
        "local_to_world_face_view": (lambda t3, r3, p3: p3),
        "local_to_world_left_wrist_view": proj_follow_left_pos_n_rot_to_head_coord,
        "local_to_world_right_wrist_view": proj_follow_right_pos_n_rot_to_head_coord,
    }


class CameraObject(ABC):
    @abstractmethod
    def actuate(self, pos, rot):
        pass

    @abstractmethod
    def reset_actuation(self):
        pass

    @abstractmethod
    def world_to_pixel(self, p3):
        pass

    @abstractmethod
    def pixel_to_world(self, p2):
        pass


class Bimanual3ViewsCamera(CameraObject):
    def __init__(
        self,
        pre_transform,
        extrinsics,
        intrinsics,
        lens_model,
        dist_coeffs,
        image_width,
        image_height,
        params_name=None,
        **kwargs,
    ):
        if params_name is None:
            params_name = {}
        self.pre_transform = pre_transform
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.lens_model = lens_model
        self.dist_coeffs = dist_coeffs
        self.image_width = image_width
        self.image_height = image_height
        self.actuated_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.actuated_rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.update_proj()
        self.pre_transform_inv = np.linalg.inv(pre_transform)
        self.extrinsics_inv = np.linalg.inv(extrinsics)
        self.params_name = params_name

    def update_proj(self):
        if self.lens_model in ["standard", "pinhole_5params"]:
            self.project_point = (
                lambda p, K=self.intrinsics, D=self.dist_coeffs: cv2.projectPoints(
                    check_points_proj(p),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    K,
                    D,
                )[0]
            )
            self.inverse_project = (
                lambda p, K=self.intrinsics, D=self.dist_coeffs: cv2.undistortPoints(
                    check_points_proj(p), K, D
                )
            )
        elif self.lens_model in ["fisheye_opencv"]:
            self.project_point = lambda p, K=self.intrinsics, D=self.dist_coeffs: cv2.fisheye.projectPoints(
                check_points_proj(p),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                K,
                D,
            )[
                0
            ]
            self.inverse_project = lambda p, K=self.intrinsics, D=self.dist_coeffs: cv2.fisheye.undistortPoints(
                check_points_proj(p), K, D
            )
        else:
            raise NotImplementedError

    def actuate(self, pos, rot):
        self.actuated_pos = np.array(pos)
        self.actuated_rot = np.array(rot)

    def reset_actuation(self):
        self.actuated_pos = npzero3()
        self.actuated_rot = npzero3()

    def world_to_pixel(self, p3):
        x = apply_T(self.pre_transform, p3)
        x = Rotation.from_euler(
            "xyz", self.actuated_rot, degrees=False
        ).as_matrix().T @ (x - self.actuated_pos)
        x = apply_T(self.extrinsics, x)
        return self.project_point(check_points_proj(x))[0, 0]

    def pixel_to_world(self, p2):
        x = self.inverse_project(p2)[0, 0]
        x = n64([x[0], x[1], 1.0])
        x = apply_T(self.extrinsics_inv, x)
        x = (
            Rotation.from_euler("xyz", self.actuated_rot, degrees=False).as_matrix() @ x
            + self.actuated_pos
        )
        return apply_T(self.pre_transform_inv, x)

    def resize_intrinsics(self, dst_wh, resize_mode="crop"):
        self.intrinsics = resize_camera_intrinsics(
            self.intrinsics, (self.image_width, self.image_height), dst_wh, resize_mode
        )
        self.image_width = dst_wh[0]
        self.image_height = dst_wh[1]
        self.update_proj()

    def serialize(self):
        name = self.params_name["camera_name"]
        intr = {
            "lens_model": self.lens_model,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": n32(self.intrinsics).tolist(),
            "dist_coeffs": n32(self.dist_coeffs).tolist(),
        }
        extr = {}
        if self.params_name["extrinsics"]["pre_transform_partial"]["valid"]:
            extr[self.params_name["extrinsics"]["pre_transform_partial"]["name"]] = n32(
                self.extrinsics @ self.pre_transform
            ).tolist()
        if self.params_name["extrinsics"]["extrinsics_partial"]["valid"]:
            extr[self.params_name["extrinsics"]["extrinsics_partial"]["name"]] = n32(
                self.extrinsics
            ).tolist()
        return (name, intr, extr)


def save_camera_groups_to_json(cameras, file_path):
    r = [camera.serialize() for camera in cameras]
    o = {
        "calibration": {
            "extrinsics": {"transform": {}},
            "intrinsics": {k: v for (k, v, _) in r},
        }
    }
    for _, __, extr in r:
        for k in extr:
            o["calibration"]["extrinsics"]["transform"][k] = extr[k]
    with open(file_path, "w") as w:
        w.write(json.dumps(o, indent=4))


@torch.no_grad()
def matrix_to_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]

    trace = torch.trace(R)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7))

    if theta < 1e-6:
        om = 0.5 * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], device=T.device
        )
    else:
        coef = theta / (2 * torch.sin(theta))
        om = coef * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], device=T.device
        )

    return torch.cat([om, t], dim=0)


def se3_to_matrix(vec6):
    w = vec6[:3]
    v = vec6[3:]
    theta = torch.norm(w)
    E = torch.eye(3, device=vec6.device)
    if theta < 1e-6:
        R = E
    else:
        axis = w / theta
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            device=vec6.device,
        )
        R = E + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    T = torch.eye(4, device=vec6.device)
    T[:3, :3] = R
    T[:3, 3] = v
    return T


@torch.no_grad()
def matrix_to_focal_centers(T):
    f1, f2, c1, c2 = T[0, 0], T[1, 1], T[0, 2], T[1, 2]
    return torch.tensor([f1, f2, c1, c2])


def focal_centers_to_matrix(vec4):
    return torch.tensor(
        [[vec4[0], 0.0, vec4[2]], [0.0, vec4[1], vec4[3]], [0.0, 0.0, 0.0]],
        dtype=vec4.dtype,
        device=vec4.device,
    )


def apply_T_torch(T, p3):
    if p3.dim() == 1:
        p3 = p3.unsqueeze(0)
    p4 = torch.cat([p3, torch.ones((p3.shape[0], 1), device=p3.device)], dim=-1)
    out = (T @ p4.T).T
    return out[:, :3]


def euler_xyz_to_matrix(rot):
    if rot.dim() == 1:
        rot = rot.unsqueeze(0)

    B = rot.shape[0]
    device = rot.device
    dtype = rot.dtype
    rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]

    sx, cx = torch.sin(rx), torch.cos(rx)
    sy, cy = torch.sin(ry), torch.cos(ry)
    sz, cz = torch.sin(rz), torch.cos(rz)

    zeros = torch.zeros(B, device=device, dtype=dtype)
    ones = torch.ones(B, device=device, dtype=dtype)

    Rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=1,
    )

    Ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=1,
    )

    Rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=1,
    )

    R = Rz @ Ry @ Rx
    return R


class Bimanual3ViewsCameraDifferentiable(torch.nn.Module):
    def __init__(self, camera):
        super().__init__()
        self.lens_model = camera.lens_model
        self.image_width = camera.image_width
        self.image_height = camera.image_height
        self.pre_transform_se3 = torch.nn.Parameter(
            matrix_to_se3(torch.from_numpy(camera.pre_transform).float())
        )
        self.extrinsics_se3 = torch.nn.Parameter(
            matrix_to_se3(torch.from_numpy(camera.extrinsics).float())
        )
        self.intrinsics_vec4 = torch.nn.Parameter(
            matrix_to_focal_centers(torch.from_numpy(camera.intrinsics).float())
        )
        self.dist_coeffs = torch.nn.Parameter(
            torch.from_numpy(camera.dist_coeffs).float()
        )
        self.pre_transform_updatable = camera.params_name["extrinsics"][
            "pre_transform_partial"
        ]["valid"]
        self.extrinsics_updatable = camera.params_name["extrinsics"][
            "extrinsics_partial"
        ]["valid"]
        self.camera = camera

    def local_to_world(self, p3, pos, rot):
        x = pos + (euler_xyz_to_matrix(rot) @ p3.unsqueeze(-1)).squeeze(-1)
        x = apply_T_torch(self.get_pre_transform_inv(), x)
        return x

    def world_to_pixel(self, p3, pos, rot):
        x = apply_T_torch(se3_to_matrix(self.pre_transform_se3), p3)
        x = (
            euler_xyz_to_matrix(rot).transpose(-1, -2) @ (x - pos).unsqueeze(-1)
        ).squeeze(-1)
        x = apply_T_torch(se3_to_matrix(self.extrinsics_se3), x)
        return self._project(x)

    def _project(self, p_cam):
        K = self.intrinsics_vec4
        D = self.dist_coeffs

        x = p_cam[:, 0] / p_cam[:, 2]
        y = p_cam[:, 1] / p_cam[:, 2]

        if self.lens_model in ["standard", "pinhole_5params"]:
            r2 = x**2 + y**2
            r4 = r2**2
            r6 = r2**3
            k1, k2, p1, p2, k3 = D[0, 0], D[1, 0], D[2, 0], D[3, 0], D[4, 0]

            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
            x_dist = x * radial + (2 * p1 * x * y + p2 * (r2 + 2 * x**2))
            y_dist = y * radial + (p1 * (r2 + 2 * y**2) + 2 * p2 * x * y)

        elif self.lens_model in ["fisheye_opencv"]:
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan(r)
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta2 * theta4
            theta8 = theta4**2
            k1, k2, k3, k4 = D[0, 0], D[1, 0], D[2, 0], D[3, 0]
            theta_d = theta * (
                1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
            )
            scale = theta_d / r
            x_dist = x * scale
            y_dist = y * scale

        else:
            raise NotImplementedError(f"Unsupported lens model: {self.lens_model}")

        u = K[0] * x_dist + K[2]
        v = K[1] * y_dist + K[3]
        return torch.stack([u, v], dim=-1)

    def get_learnable_parameters(self):
        params = []
        params.append({"params": [self.intrinsics_vec4], "lr": 1e-2})
        params.append({"params": [self.dist_coeffs], "lr": 1e-5})
        if self.extrinsics_updatable:
            params.append({"params": [self.extrinsics_se3], "lr": 1e-3})
        if self.pre_transform_updatable:
            params.append({"params": [self.pre_transform_se3], "lr": 1e-3})
        return params

    def get_pre_transform_inv(self):
        T = se3_to_matrix(self.pre_transform_se3)
        # 假设 T 是 (B, 4, 4) 或 (4, 4)
        R = T[..., :3, :3]
        t = T[..., :3, 3:4]
        R_inv = R.transpose(-1, -2)
        t_inv = -torch.matmul(R_inv, t)
        inv_T = torch.eye(4, device=T.device, dtype=T.dtype).expand_as(T).clone()
        inv_T[..., :3, :3] = R_inv
        inv_T[..., :3, 3:4] = t_inv

        return inv_T

    @torch.no_grad()
    def serialize(self):
        name = self.camera.params_name["camera_name"]
        intr = {
            "lens_model": self.lens_model,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": n32(
                focal_centers_to_matrix(self.intrinsics_vec4).cpu()
            ).tolist(),
            "dist_coeffs": n32(self.dist_coeffs.cpu()).tolist(),
        }
        extr = {}
        if self.camera.params_name["extrinsics"]["pre_transform_partial"]["valid"]:
            extr[
                self.camera.params_name["extrinsics"]["pre_transform_partial"]["name"]
            ] = n32(
                se3_to_matrix(self.extrinsics_se3).cpu().numpy()
                @ se3_to_matrix(self.pre_transform_se3).cpu().numpy()
            ).tolist()
        if self.camera.params_name["extrinsics"]["extrinsics_partial"]["valid"]:
            extr[
                self.camera.params_name["extrinsics"]["extrinsics_partial"]["name"]
            ] = n32(se3_to_matrix(self.extrinsics_se3).cpu()).tolist()
        return (name, intr, extr)


class SightCones(ABC):
    @abstractmethod
    def actuated_cones(self, pos, rot):
        pass

    @abstractmethod
    def batched_actuated_cones(self, poses, rots):
        pass

    @staticmethod
    @abstractmethod
    def intersection(
        *args, **kwargs
    ):  # return an intersection map (True / False) for a batch of cones with each other
        pass


class BimanualCameraSightCones(SightCones):
    def __init__(
        self, camera, patch_to_pixel_size=16, **kwargs
    ):  # H W D, D = 7: [x0, y0, z0, u, v, w, r]
        # Cone L : d((x, y, z), (x0, y0, z0) + (u, v, w) * t) = r * t
        self.camera = camera
        self.patch_to_pixel_size = patch_to_pixel_size
        self.width = camera.image_width // self.patch_to_pixel_size
        self.height = camera.image_height // self.patch_to_pixel_size
        self.token_count = self.width * self.height
        self.default_cones = np.zeros((self.height, self.width, 7), dtype=np.float64)
        self.camera.reset_actuation()
        origin = apply_T(
            self.camera.pre_transform_inv,
            apply_T(self.camera.extrinsics_inv, npzero3()),
        )
        self.w1 = torch.tensor(self.camera.pre_transform[:3, :3], dtype=torch.float32)
        self.b1 = torch.tensor(self.camera.pre_transform[:3, 3], dtype=torch.float32)[
            None, None, :
        ]
        self.w2 = torch.tensor(
            self.camera.pre_transform_inv[:3, :3], dtype=torch.float32
        )
        self.b2 = torch.tensor(
            self.camera.pre_transform_inv[:3, 3], dtype=torch.float32
        )[None, None, :]
        for i in range(self.width):
            for j in range(self.height):
                center = (
                    (i + 0.5) * self.patch_to_pixel_size,
                    (j + 0.5) * self.patch_to_pixel_size,
                )
                target = self.camera.pixel_to_world(center)
                self.default_cones[j, i, :3] = origin
                u = nvec(target - origin)
                self.default_cones[j, i, 3:6] = u
                tans = []
                for lr in [-0.5, 0.5]:
                    for td in [-0.5, 0.5]:
                        corner = (
                            (i + 0.5 + lr) * self.patch_to_pixel_size,
                            (j + 0.5 + td) * self.patch_to_pixel_size,
                        )
                        corner = self.camera.pixel_to_world(corner)
                        v = nvec(corner - origin)
                        cos = np.sum(u * v)
                        tan = np.sqrt(1 - cos**2) / cos
                        tans.append(tan)
                self.default_cones[j, i, 6] = np.max(tans)
        self.default_cones = torch.tensor(self.default_cones, dtype=torch.float32)

    def actuated_cones(self, pos, rot):
        r = torch.tensor(
            Rotation.from_euler(
                "xyz", _to_numpy_float32(rot), degrees=False
            ).as_matrix()
        ).to(dtype=torch.float32, device=self.device)
        t = torch.tensor(_to_numpy_float32(pos)).to(
            dtype=torch.float32, device=self.device
        )[None, None, :]
        o = torch.einsum("mn,hwn->hwm", self.w1, self.default_cones[:, :, :3]) + self.b1
        o = torch.einsum("mn,hwn->hwm", r, o) + t
        o = torch.einsum("mn,hwn->hwm", self.w2, o) + self.b2
        u = torch.einsum("mn,hwn->hwm", self.w1, self.default_cones[:, :, 3:6])
        u = torch.einsum("mn,hwn->hwm", r, u)
        u = torch.einsum("mn,hwn->hwm", self.w2, u)
        return torch.cat([o, u, self.default_cones[:, :, 6:]], dim=-1)

    @torch.no_grad()
    def batched_actuated_cones(self, poses, rots):
        poses_np = _to_numpy_float32(poses)
        rots_np = _to_numpy_float32(rots)
        r = torch.tensor(
            np.array(
                [
                    Rotation.from_euler("xyz", rot, degrees=False).as_matrix()
                    for rot in rots_np
                ]
            )
        ).to(dtype=torch.float32, device=self.device)
        t = torch.tensor(poses_np).to(dtype=torch.float32, device=self.device)[
            :, None, None, :
        ]
        o = torch.einsum("mn,hwn->hwm", self.w1, self.default_cones[:, :, :3]) + self.b1
        o = torch.einsum("tmn,hwn->thwm", r, o) + t
        o = torch.einsum("mn,thwn->thwm", self.w2, o) + self.b2[None, :, :, :]
        u = torch.einsum("mn,hwn->hwm", self.w1, self.default_cones[:, :, 3:6])
        u = torch.einsum("tmn,hwn->thwm", r, u)
        u = torch.einsum("mn,thwn->thwm", self.w2, u)
        return torch.cat(
            [
                o,
                u,
                self.default_cones[:, :, 6:][None, :, :, :].repeat(o.shape[0], 1, 1, 1),
            ],
            dim=-1,
        )

    @staticmethod
    @torch.no_grad
    def intersection(
        cones, max_len=5.0, min_len=0.2, eps=1e-6, radius_multiplier=1.0
    ):  # [B N D], D=7
        P = cones[:, :, 0:3]
        D = cones[:, :, 3:6]
        K = cones[:, :, 6] * radius_multiplier
        P1, P2 = P[:, :, None, :], P[:, None, :, :]
        D1, D2 = D[:, :, None, :], D[:, None, :, :]
        K1, K2 = K[:, :, None], K[:, None, :]
        v = P1 - P2
        b = (D1 * D2).sum(dim=-1)
        d = (D1 * v).sum(dim=-1)
        e = (D2 * v).sum(dim=-1)
        denom = 1.0 - b * b
        safe_denom = torch.where(denom > eps, denom, torch.ones_like(denom))
        t1 = torch.where(denom > eps, (-d + b * e) / safe_denom, torch.zeros_like(d))
        # t2 = torch.where(denom > eps, (e - b * d) / safe_denom, e)
        t1 = torch.clamp(t1, min_len, max_len)
        t2 = torch.clamp(t1 * b + e, min_len, max_len)
        t1 = torch.clamp(t2 * b - d, min_len, max_len)
        # Two iter truncation
        C1 = P1 + t1.unsqueeze(-1) * D1
        C2 = P2 + t2.unsqueeze(-1) * D2
        dist = torch.norm(C1 - C2, dim=-1)
        R1 = t1 * K1
        R2 = t2 * K2
        intersect_mask = dist <= (R1 + R2)
        return intersect_mask

    def to(self, *args, **kwargs):
        self.default_cones = self.default_cones.to(*args, **kwargs)
        self.w1 = self.w1.to(*args, **kwargs)
        self.w2 = self.w2.to(*args, **kwargs)
        self.b1 = self.b1.to(*args, **kwargs)
        self.b2 = self.b2.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self.default_cones.device


def bimanual_3views_actuation(cones, action, view_map):
    action = action.cpu()
    fetch_to_pos_rot = {
        "face_view": (
            lambda x: (torch.zeros_like(x[:, :3]), torch.zeros_like(x[:, :3]))
        ),
        "left_wrist_view": (lambda x: (x[:, :3], x[:, 3:6])),
        "right_wrist_view": (lambda x: (x[:, 7:10], x[:, 10:13])),
    }
    r = {}
    for view_name in view_map:
        target = view_map[view_name]
        pos, rot = fetch_to_pos_rot[target](action)
        r[view_name] = cones[view_name].batched_actuated_cones(pos, rot)
    return r


MAPPING_DATASET_TO_CAMERA = {"x2_normal": read_cameras_for_bimanual_3views}

MAPPING_DATASET_TO_SIGHTCONE = {"x2_normal": BimanualCameraSightCones}

MAPPING_ACTUATIONS = {"x2_normal": bimanual_3views_actuation}


### Example camera information json file for robot_id=10071
# {
#     "calibration": {
#         "extrinsics": {
#             "transform": {
#                 "M_left_to_head": [
#                     [
#                         0.9985812038341202,
#                         -0.014677648827356997,
#                         0.051187361468434715,
#                         0.30165335269725774
#                     ],
#                     [
#                         -0.013845009417782064,
#                         0.8566397069141239,
#                         0.5157293168439288,
#                         -0.2923215517819286
#                     ],
#                     [
#                         -0.051418820128635155,
#                         -0.5157062915681607,
#                         0.855221097596168,
#                         -0.2197971725396255
#                     ],
#                     [
#                         0,
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "M_right_to_head": [
#                     [
#                         0.9962758379512912,
#                         0.03876369480554731,
#                         0.07701837884216274,
#                         -0.3216673587050007
#                     ],
#                     [
#                         -0.07344480226954411,
#                         0.8494635358465417,
#                         0.5225108250425811,
#                         -0.2822252070211936
#                     ],
#                     [
#                         -0.0451698542618867,
#                         -0.5262215096631014,
#                         0.8491469878848104,
#                         -0.18807321999526672
#                     ],
#                     [
#                         0,
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "M_gripper_to_camL": [
#                     [
#                         0.004326625390726396,
#                         -0.9995914106049851,
#                         -0.028254064441502524,
#                         -0.0015168979843534734
#                     ],
#                     [
#                         -0.4449971687956976,
#                         0.023378011101523294,
#                         -0.8952267804086016,
#                         0.12043077671837227
#                     ],
#                     [
#                         0.8955215240721699,
#                         0.016446289602013886,
#                         -0.4447141997752996,
#                         -0.04019246835084772
#                     ],
#                     [
#                         0,
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "M_gripper_to_camR": [
#                     [
#                         -0.017392597584590608,
#                         -0.9975063245607478,
#                         -0.06840051177124684,
#                         -0.004864993763888995
#                     ],
#                     [
#                         -0.4713423537366601,
#                         0.06851227258283016,
#                         -0.8792851949620903,
#                         0.12096722019473426
#                     ],
#                     [
#                         0.881778817574592,
#                         0.016947004656987674,
#                         -0.47135858527230656,
#                         -0.0430900121102918
#                     ],
#                     [
#                         0,
#                         0,
#                         0,
#                         1
#                     ]
#                 ]
#             }
#         },
#         "intrinsics": {
#             "head": {
#                 "date": "2025-12-26 14:43:32",
#                 "lens_model": "standard",
#                 "image_width": 1280,
#                 "image_height": 960,
#                 "camera_matrix": [
#                     [
#                         956.9476846742253,
#                         0,
#                         641.2433926678578
#                     ],
#                     [
#                         0,
#                         956.9673120652931,
#                         486.9126083677085
#                     ],
#                     [
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "dist_coeffs": [
#                     [
#                         0.0024427410912930513
#                     ],
#                     [
#                         -0.03920373113595619
#                     ],
#                     [
#                         0.0008456926549719285
#                     ],
#                     [
#                         -0.00007119841081130314
#                     ],
#                     [
#                         0
#                     ]
#                 ],
#                 "reprojection_error": 0.16226830061379163
#             },
#             "left": {
#                 "date": "2025-12-26 14:32:41",
#                 "lens_model": "fisheye_opencv",
#                 "image_width": 1920,
#                 "image_height": 1080,
#                 "camera_matrix": [
#                     [
#                         642.5229262567373,
#                         0,
#                         961.4649881994591
#                     ],
#                     [
#                         0,
#                         642.4127223245516,
#                         559.4344390857693
#                     ],
#                     [
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "dist_coeffs": [
#                     [
#                         -0.002787018873692755
#                     ],
#                     [
#                         0.000079562081049256
#                     ],
#                     [
#                         -0.0038728902459226034
#                     ],
#                     [
#                         0.0010598420688560568
#                     ]
#                 ],
#                 "reprojection_error": 0.2253333762189471
#             },
#             "right": {
#                 "date": "2025-12-26 14:25:58",
#                 "lens_model": "fisheye_opencv",
#                 "image_width": 1920,
#                 "image_height": 1080,
#                 "camera_matrix": [
#                     [
#                         655.7970820210048,
#                         0,
#                         930.0394827772064
#                     ],
#                     [
#                         0,
#                         657.6462689226,
#                         549.4825169719243
#                     ],
#                     [
#                         0,
#                         0,
#                         1
#                     ]
#                 ],
#                 "dist_coeffs": [
#                     [
#                         -0.008241095805724385
#                     ],
#                     [
#                         -0.0006318967964794762
#                     ],
#                     [
#                         0.0046734019860027
#                     ],
#                     [
#                         -0.0031082899915000984
#                     ]
#                 ],
#                 "reprojection_error": 0.6412805893544744
#             }
#         }
#     }
# }
