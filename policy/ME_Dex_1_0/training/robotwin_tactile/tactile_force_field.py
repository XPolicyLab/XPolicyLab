"""Aligned simulated TacFF collection for RoboTwin replay.

The collector is deliberately independent of the policy/model representation:
it accumulates contact impulses at the physics rate and flushes one dense force
field whenever the RGB observation cache is flushed.
"""
from __future__ import annotations

from collections import deque
import json
import hashlib
from pathlib import Path

import numpy as np


DEFAULT_PAD_NAMES = ("fl_link7", "fl_link8", "fr_link7", "fr_link8")

# Each opposing ALOHA pad uses a mirrored local-Y axis.  These signs were
# verified on the existing clean-50 replay: link7 receives compressive force
# along +Y and link8 along -Y on both grippers.  Tangential u/v follow the
# grid's increasing local +X/+Z coordinates.
DEFAULT_NORMAL_SIGNS = {
    "fl_link7": 1.0,
    "fl_link8": -1.0,
    "fr_link7": 1.0,
    "fr_link8": -1.0,
}


class RegularMaskedSurfaceAtlas:
    """Geometry-derived TFA-2.0 grid on the gripper collision surface."""

    def __init__(self, mesh_path, normal_sign, *, height=10, width=14,
                 x_range=(-0.005, 0.070), z_range=(-0.016, 0.011)):
        self.mesh_path = Path(mesh_path)
        self.normal_sign = float(normal_sign)
        self.height, self.width = int(height), int(width)
        self.x_range = tuple(float(v) for v in x_range)
        self.z_range = tuple(float(v) for v in z_range)
        self.vertices, self.normals = self._read_binary_stl(self.mesh_path)
        self.mesh_sha256 = hashlib.sha256(self.mesh_path.read_bytes()).hexdigest()
        self.x = np.linspace(*self.x_range, self.width, dtype=np.float64)
        self.z = np.linspace(*self.z_range, self.height, dtype=np.float64)
        self.node_xyz, self.triangle_id, self.barycentric = self._intersect_nodes()
        self.mask = self.triangle_id >= 0

    @staticmethod
    def _read_binary_stl(path):
        blob = path.read_bytes()
        if len(blob) < 84:
            raise ValueError(f"Invalid binary STL: {path}")
        count = int(np.frombuffer(blob, dtype="<u4", count=1, offset=80)[0])
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                          ("attribute", "<u2")])
        expected = 84 + count * dtype.itemsize
        if len(blob) != expected:
            raise ValueError(f"Expected {expected} bytes for {count} STL triangles, got {len(blob)}")
        records = np.frombuffer(blob, dtype=dtype, count=count, offset=84)
        return records["vertices"].astype(np.float64), records["normal"].astype(np.float64)

    def _intersect_nodes(self):
        xyz = np.full((self.height, self.width, 3), np.nan, dtype=np.float64)
        triangle_id = np.full((self.height, self.width), -1, dtype=np.int32)
        barycentric = np.full((self.height, self.width, 3), np.nan, dtype=np.float64)
        candidates = np.flatnonzero(self.normals[:, 1] * self.normal_sign > 0.5)
        eps = 1e-8
        for row, z in enumerate(self.z):
            for col, x in enumerate(self.x):
                best = None
                for index in candidates:
                    tri = self.vertices[index]
                    x1, z1 = tri[0, (0, 2)]
                    x2, z2 = tri[1, (0, 2)]
                    x3, z3 = tri[2, (0, 2)]
                    denominator = (z2-z3)*(x1-x3) + (x3-x2)*(z1-z3)
                    if abs(denominator) <= eps:
                        continue
                    a = ((z2-z3)*(x-x3) + (x3-x2)*(z-z3)) / denominator
                    b = ((z3-z1)*(x-x3) + (x1-x3)*(z-z3)) / denominator
                    c = 1.0 - a - b
                    if min(a, b, c) < -eps:
                        continue
                    score = self.normals[index, 1] * self.normal_sign
                    if best is None or score > best[0]:
                        best = (score, index, np.array([a, b, c]))
                if best is not None:
                    _, index, weights = best
                    xyz[row, col] = weights @ self.vertices[index]
                    triangle_id[row, col] = index
                    barycentric[row, col] = weights
        return xyz, triangle_id, barycentric

    def splat_neighbors(self, x, z):
        """Return mask-renormalized bilinear neighbors; never boundary-clamp."""
        u = (float(x) - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        v = (float(z) - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return []
        gx, gy = u * (self.width - 1), v * (self.height - 1)
        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        candidates = []
        for iy, ix, weight in (
            (y0, x0, (1-(gx-x0))*(1-(gy-y0))),
            (y0, x0+1, (gx-x0)*(1-(gy-y0))),
            (y0+1, x0, (1-(gx-x0))*(gy-y0)),
            (y0+1, x0+1, (gx-x0)*(gy-y0)),
        ):
            if (0 <= iy < self.height and 0 <= ix < self.width and
                    self.mask[iy, ix] and weight > 0):
                candidates.append((iy, ix, weight))
        total = sum(item[2] for item in candidates)
        return [(iy, ix, weight / total) for iy, ix, weight in candidates] if total > 0 else []


def _canonical_transform(normal_sign):
    return np.array([
        [0.0, float(normal_sign), 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _quat_to_matrix_wxyz(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TactileForceFieldCollector:
    """Rasterize SAPIEN contact impulses to four link-local 10x14 TacFF pads.

    The rectangular sensor bounds are intentionally explicit calibration
    parameters.  The first pilot also records out-of-bounds contacts, so we do
    not hide a bad link frame or an undersized active area by clamping points.
    """

    def __init__(self, robot, scene, timestep, save_dir, *, height=10, width=14,
                 pad_names=None, sensor_length=0.040, sensor_width=0.028,
                 surface_axes=(0, 2), surface_bounds=((-0.010, 0.090), (-0.015, 0.035)),
                 normal_signs=None, min_normal_alignment=0.0, spec_version="TFA-1.0",
                 mesh_paths=None, grid_x_range=(-0.005, 0.070),
                 grid_z_range=(-0.016, 0.011), native_frame_physics_steps=None,
                 completed_frame_queue_size=64):
        self.robot = robot
        self.scene = scene
        self.timestep = float(timestep)
        self.height = int(height)
        self.width = int(width)
        self.pad_names = tuple(DEFAULT_PAD_NAMES if pad_names is None else pad_names)
        self.sensor_length = float(sensor_length)
        self.sensor_width = float(sensor_width)
        self.surface_axes = tuple(int(axis) for axis in surface_axes)
        self.surface_bounds = np.asarray(surface_bounds, dtype=np.float64)
        if normal_signs is None:
            unknown = [name for name in self.pad_names if name not in DEFAULT_NORMAL_SIGNS]
            if unknown:
                raise ValueError(
                    "normal_signs is required for non-ALOHA tactile pads: "
                    f"{unknown}"
                )
            normal_signs = [DEFAULT_NORMAL_SIGNS[name] for name in self.pad_names]
        if len(normal_signs) != len(self.pad_names):
            raise ValueError("normal_signs must contain one entry per tactile pad")
        self.normal_signs = tuple(float(sign) for sign in normal_signs)
        if any(abs(sign) != 1.0 for sign in self.normal_signs):
            raise ValueError("Every tactile-pad normal sign must be +1 or -1")
        self._canonical_transforms = np.stack([
            _canonical_transform(sign) for sign in self.normal_signs
        ])
        self.min_normal_alignment = float(min_normal_alignment)
        self.spec_version = str(spec_version)
        self.native_frame_physics_steps = (
            None
            if native_frame_physics_steps is None
            else int(native_frame_physics_steps)
        )
        self.completed_frame_queue_size = int(completed_frame_queue_size)
        if self.native_frame_physics_steps is not None:
            if self.spec_version != "TFA-2.0":
                raise ValueError(
                    "Native tactile framing is only supported for spec_version=TFA-2.0"
                )
            if self.native_frame_physics_steps <= 0:
                raise ValueError("native_frame_physics_steps must be positive")
            if self.completed_frame_queue_size < 2:
                raise ValueError("completed_frame_queue_size must be at least two")
        if not -1.0 <= self.min_normal_alignment <= 1.0:
            raise ValueError("min_normal_alignment must be in [-1, 1]")
        links = list(robot.left_entity.get_links()) + list(robot.right_entity.get_links())
        by_name = {link.get_name(): link for link in links}
        missing = [name for name in self.pad_names if name not in by_name]
        if missing:
            raise RuntimeError(f"TacFF pad links not found: {missing}; available={sorted(by_name)}")
        self.pads = [by_name[name] for name in self.pad_names]
        self._pad_index = {link.get_name(): idx for idx, link in enumerate(self.pads)}
        self.atlases = None
        if self.spec_version == "TFA-2.0":
            if mesh_paths is None or len(mesh_paths) != len(self.pad_names):
                raise ValueError("TFA-2.0 requires one collision mesh_path per tactile pad")
            self.atlases = [RegularMaskedSurfaceAtlas(
                path, sign, height=self.height, width=self.width,
                x_range=grid_x_range, z_range=grid_z_range,
            ) for path, sign in zip(mesh_paths, self.normal_signs)]
        self._total_physics_steps = 0
        self._completed_frames = deque(maxlen=self.completed_frame_queue_size)
        self._write_metadata(save_dir)
        self.reset_interval()

    def _write_metadata(self, save_dir):
        path = Path(save_dir) / "tactile_force_field_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "tactile_force_field_v2",
            "spec_version": self.spec_version,
            "shape_per_frame": [len(self.pads), 3, self.height, self.width],
            "pad_names": list(self.pad_names),
            "physics_timestep_seconds": self.timestep,
            "native_frame_physics_steps": self.native_frame_physics_steps,
            "native_frame_interval_seconds": (
                self.native_frame_physics_steps * self.timestep
                if self.native_frame_physics_steps is not None
                else None
            ),
            "surface_axes_link_local": list(self.surface_axes),
            "surface_bounds_link_local_m": self.surface_bounds.tolist(),
            "channels": ["Fx_link", "Fy_link", "Fz_link"],
            "canonical_channels": ["normal", "shear_u", "shear_v"],
            "canonical_axes_link_local_by_pad": {
                name: {
                    "normal": [0.0, sign, 0.0],
                    "shear_u": [1.0, 0.0, 0.0],
                    "shear_v": [0.0, 0.0, 1.0],
                }
                for name, sign in zip(self.pad_names, self.normal_signs)
            },
            "canonical_axis_handedness_by_pad": {
                name: int(np.sign(np.linalg.det(transform)))
                for name, transform in zip(
                    self.pad_names, self._canonical_transforms
                )
            },
            "canonical_units": "newton",
            "front_face_filter": {
                "source": "signed SAPIEN contact-point normal in pad-local frame",
                "minimum_normal_alignment": self.min_normal_alignment,
                "rule": "dot(contact_normal_on_pad, canonical_normal) > minimum",
            },
            "aggregation": "sum(contact_impulse)/elapsed_seconds; bilinear spatial splat",
            "contact_sign_convention": "SAPIEN point.impulse is assigned to bodies[0]; negated for bodies[1]",
            "calibration_status": "pilot; inspect out_of_bounds and contact diagnostics before production",
        }
        if self.atlases is not None:
            metadata["regular_masked_atlas"] = {
                "x_range_m": list(self.atlases[0].x_range),
                "z_range_m": list(self.atlases[0].z_range),
                "valid_nodes_by_pad": [int(atlas.mask.sum()) for atlas in self.atlases],
                "mesh_sha256_by_pad": [atlas.mesh_sha256 for atlas in self.atlases],
                "support_mask_by_pad": [atlas.mask.astype(np.uint8).tolist() for atlas in self.atlases],
                "node_xyz_link_local_m_by_pad": [
                    np.where(atlas.mask[..., None], atlas.node_xyz, 0.0).tolist()
                    for atlas in self.atlases
                ],
                "triangle_id_by_pad": [atlas.triangle_id.tolist() for atlas in self.atlases],
                "barycentric_by_pad": [
                    np.where(atlas.mask[..., None], atlas.barycentric, 0.0).tolist()
                    for atlas in self.atlases
                ],
                "mask_rule": "front-facing ray hit against fixed collision STL",
                "splat_rule": "masked bilinear; invalid neighbors removed and weights renormalized; no clamping",
            }
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _to_canonical(link_xyz, transforms):
        """Map link-local XYZ vectors to [normal, shear_u, shear_v]."""
        values = np.asarray(link_xyz)
        transforms = np.asarray(transforms)
        if values.ndim != 4 or values.shape[1] != 3:
            raise ValueError(
                "Expected [pads, 3, height, width], "
                f"got shape {values.shape}"
            )
        if transforms.shape != (values.shape[0], 3, 3):
            raise ValueError(
                f"Expected transforms [{values.shape[0]}, 3, 3], "
                f"got {transforms.shape}"
            )
        return np.einsum("pij,pjhw->pihw", transforms, values)

    def reset_interval(self):
        self._impulse = np.zeros((len(self.pads), 3, self.height, self.width), dtype=np.float64)
        self._net_impulse_all = np.zeros((len(self.pads), 3), dtype=np.float64)
        self._net_impulse_grid = np.zeros((len(self.pads), 3), dtype=np.float64)
        self._contact_count = np.zeros((len(self.pads),), dtype=np.int32)
        self._raw_contact_count = np.zeros((len(self.pads),), dtype=np.int32)
        self._rejected_backface_count = np.zeros((len(self.pads),), dtype=np.int32)
        self._rejected_backface_impulse = np.zeros((len(self.pads), 3), dtype=np.float64)
        self._normal_alignment_min = np.full((len(self.pads),), np.inf, dtype=np.float64)
        self._normal_alignment_max = np.full((len(self.pads),), -np.inf, dtype=np.float64)
        self._out_of_bounds = np.zeros((len(self.pads),), dtype=np.int32)
        self._local_position_min = np.full((len(self.pads), 3), np.inf, dtype=np.float64)
        self._local_position_max = np.full((len(self.pads), 3), -np.inf, dtype=np.float64)
        self._physics_steps = 0
        self._interval_start_physics_step = self._total_physics_steps + 1

    def _link_local(self, link, world_position, world_vector):
        pose = link.get_pose()
        rot = _quat_to_matrix_wxyz(pose.q)
        local_position = rot.T @ (np.asarray(world_position, dtype=np.float64) - pose.p)
        local_vector = rot.T @ np.asarray(world_vector, dtype=np.float64)
        return local_position, local_vector

    def _front_face_alignment(self, pad_index, local_contact_normal):
        normal = np.asarray(local_contact_normal, dtype=np.float64)
        magnitude = np.linalg.norm(normal)
        if magnitude <= 1e-12:
            return -np.inf
        canonical_normal = self._canonical_transforms[pad_index, 0]
        return float(np.dot(normal / magnitude, canonical_normal))

    def _splat(self, pad_index, position, impulse):
        if self.atlases is not None:
            neighbors = self.atlases[pad_index].splat_neighbors(position[0], position[2])
            if not neighbors:
                self._out_of_bounds[pad_index] += 1
                return False
            for iy, ix, weight in neighbors:
                self._impulse[pad_index, :, iy, ix] += weight * impulse
            return True
        # Link-local x/y form the provisional active rectangle.  z is retained
        # in the force vector but is not used as a spatial coordinate.
        u = (position[self.surface_axes[0]] - self.surface_bounds[0, 0]) / (self.surface_bounds[0, 1] - self.surface_bounds[0, 0])
        v = (position[self.surface_axes[1]] - self.surface_bounds[1, 0]) / (self.surface_bounds[1, 1] - self.surface_bounds[1, 0])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            self._out_of_bounds[pad_index] += 1
            return False
        x = u * (self.width - 1)
        y = v * (self.height - 1)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)
        wx, wy = x - x0, y - y0
        for iy, ix, weight in (
            (y0, x0, (1 - wx) * (1 - wy)),
            (y0, x1, wx * (1 - wy)),
            (y1, x0, (1 - wx) * wy),
            (y1, x1, wx * wy),
        ):
            self._impulse[pad_index, :, iy, ix] += weight * impulse
        return True

    def record_physics_step(self):
        """Collect contacts after exactly one ``scene.step()``."""
        self._total_physics_steps += 1
        self._physics_steps += 1
        for contact in self.scene.get_contacts():
            bodies = contact.bodies
            for body_index, body in enumerate(bodies):
                link = getattr(body, "entity", None)
                pad_index = self._pad_index.get(link.get_name()) if link is not None else None
                if pad_index is None:
                    continue
                # SAPIEN defines the point impulse on bodies[0]; apply the
                # equal-and-opposite impulse when this pad is bodies[1].
                sign = 1.0 if body_index == 0 else -1.0
                for point in contact.points:
                    local_position, local_impulse = self._link_local(
                        link, point.position, sign * point.impulse)
                    _, local_contact_normal = self._link_local(
                        link, point.position, sign * point.normal)
                    alignment = self._front_face_alignment(
                        pad_index, local_contact_normal
                    )
                    self._raw_contact_count[pad_index] += 1
                    self._normal_alignment_min[pad_index] = min(
                        self._normal_alignment_min[pad_index], alignment
                    )
                    self._normal_alignment_max[pad_index] = max(
                        self._normal_alignment_max[pad_index], alignment
                    )
                    self._local_position_min[pad_index] = np.minimum(self._local_position_min[pad_index], local_position)
                    self._local_position_max[pad_index] = np.maximum(self._local_position_max[pad_index], local_position)
                    if alignment <= self.min_normal_alignment:
                        self._rejected_backface_count[pad_index] += 1
                        self._rejected_backface_impulse[pad_index] += local_impulse
                        continue
                    if self._splat(pad_index, local_position, local_impulse):
                        self._net_impulse_grid[pad_index] += local_impulse
                    self._net_impulse_all[pad_index] += local_impulse
                    self._contact_count[pad_index] += 1

        if (
            self.native_frame_physics_steps is not None
            and self._physics_steps == self.native_frame_physics_steps
        ):
            self._completed_frames.append(self._finalize_frame())

    @property
    def total_physics_steps(self):
        """Number of physics steps observed since collector construction."""
        return self._total_physics_steps

    @property
    def completed_frame_count(self):
        return len(self._completed_frames)

    def seed_zero_bootstrap_frame(self):
        """Seed the episode-boundary frame without advancing simulation time."""
        if self.native_frame_physics_steps is None or self.spec_version != "TFA-2.0":
            raise RuntimeError("Zero bootstrap is only valid for native TFA-2.0 framing")
        if self._total_physics_steps != 0 or self._physics_steps != 0:
            raise RuntimeError("Zero bootstrap must be created before recorded physics steps")
        if self._completed_frames:
            raise RuntimeError("TFA-2.0 bootstrap frame is already present")
        payload = self._build_frame_payload()
        payload["start_physics_step"] = np.int64(0)
        payload["end_physics_step"] = np.int64(0)
        payload["is_bootstrap"] = True
        self._completed_frames.append(payload)

    @staticmethod
    def _copy_payload(payload):
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in payload.items()
        }

    def get_completed_frames(self, count=2, *, up_to_physics_step=None):
        """Return the newest complete native frames without consuming the FIFO."""
        if self.native_frame_physics_steps is None:
            raise RuntimeError("Collector is not configured for native tactile framing")
        count = int(count)
        if count <= 0:
            raise ValueError("count must be positive")
        observation_step = (
            self._total_physics_steps
            if up_to_physics_step is None
            else int(up_to_physics_step)
        )
        if observation_step > self._total_physics_steps:
            raise ValueError(
                "up_to_physics_step cannot be later than the collector clock: "
                f"{observation_step} > {self._total_physics_steps}"
            )
        eligible = [
            payload
            for payload in self._completed_frames
            if int(payload["end_physics_step"]) <= observation_step
        ]
        return [self._copy_payload(payload) for payload in eligible[-count:]]

    def _build_frame_payload(self):
        """Build one frame from the current interval without resetting it."""
        elapsed = self._physics_steps * self.timestep
        if elapsed > 0:
            field = (self._impulse / elapsed).astype(np.float32)
            net_force_all = (self._net_impulse_all / elapsed).astype(np.float32)
            net_force_grid = (self._net_impulse_grid / elapsed).astype(np.float32)
        else:
            field = self._impulse.astype(np.float32)
            net_force_all = self._net_impulse_all.astype(np.float32)
            net_force_grid = self._net_impulse_grid.astype(np.float32)
        force_canonical = self._to_canonical(
            field, self._canonical_transforms
        ).astype(np.float32)
        net_force_all_canonical = np.einsum(
            "pij,pj->pi", self._canonical_transforms, net_force_all
        ).astype(np.float32)
        net_force_grid_canonical = np.einsum(
            "pij,pj->pi", self._canonical_transforms, net_force_grid
        ).astype(np.float32)
        payload = {
            "force": field,
            # Compatibility aliases retain the original all-contact meaning.
            "net_force": net_force_all,
            "force_canonical": force_canonical,
            "net_force_canonical": net_force_all_canonical,
            "net_force_all": net_force_all,
            "net_force_grid": net_force_grid,
            "net_force_all_canonical": net_force_all_canonical,
            "net_force_grid_canonical": net_force_grid_canonical,
            "contact_count": self._contact_count.copy(),
            "raw_contact_count": self._raw_contact_count.copy(),
            "rejected_backface_count": self._rejected_backface_count.copy(),
            "rejected_backface_impulse": self._rejected_backface_impulse.astype(np.float32),
            "normal_alignment_min": self._normal_alignment_min.astype(np.float32),
            "normal_alignment_max": self._normal_alignment_max.astype(np.float32),
            "out_of_bounds": self._out_of_bounds.copy(),
            "local_position_min": self._local_position_min.astype(np.float32),
            "local_position_max": self._local_position_max.astype(np.float32),
            "physics_steps": np.int32(self._physics_steps),
            "interval_seconds": np.float64(elapsed),
            "start_physics_step": np.int64(self._interval_start_physics_step),
            "end_physics_step": np.int64(self._total_physics_steps),
            "is_bootstrap": False,
        }
        if self.atlases is not None:
            # Repeated per observation for self-contained HDF5 episodes.  The
            # array is tiny (4x10x14 uint8) and makes missing != zero explicit.
            payload["support_mask"] = np.stack(
                [atlas.mask for atlas in self.atlases]
            ).astype(np.uint8)
        return payload

    def _finalize_frame(self):
        payload = self._build_frame_payload()
        self.reset_interval()
        return payload

    def flush_frame(self):
        """Flush one caller-defined interval for offline dataset generation.

        Closed-loop TFA-2.0 uses the native FIFO and must never flush a partial
        action-dependent interval.
        """
        if self.native_frame_physics_steps is not None:
            raise RuntimeError(
                "flush_frame() is disabled for native TFA-2.0 framing; "
                "use get_completed_frames()"
            )
        return self._finalize_frame()
