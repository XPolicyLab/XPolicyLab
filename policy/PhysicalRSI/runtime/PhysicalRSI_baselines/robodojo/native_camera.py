"""Reset-only camera SDK diagnostics; never a policy input or renderer qualification."""

import numpy as np

from .camera_geometry import PinholeCamera


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite native camera metadata")
    return result


def inspect_camera(camera):
    width, height = map(int, camera.get_resolution())
    position, orientation = camera.get_world_pose(camera_axes="usd")
    intrinsics = array(camera.get_intrinsics_matrix())
    view = array(camera.get_view_matrix_ros())
    if intrinsics.shape != (3, 3) or view.shape != (4, 4):
        raise ValueError("Invalid SDK camera matrix shape")
    expected = np.array(
        [
            [intrinsics[0, 0], 0, intrinsics[0, 2]],
            [0, intrinsics[1, 1], intrinsics[1, 2]],
            [0, 0, 1],
        ]
    )
    if not np.allclose(intrinsics, expected, atol=1e-6, rtol=0):
        raise ValueError("SDK camera must use zero-skew pinhole intrinsics")
    configuration = dict(
        width=width,
        height=height,
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
        world_pose=[*array(position).tolist(), *array(orientation).tolist()],
    )
    pinhole = PinholeCamera(**configuration)
    pixels = np.array(
        [
            [width * u, height * v]
            for u, v in [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]
        ]
    )
    points = pinhole.unproject(pixels, np.array([0.3, 0.5, 1.0, 1.5, 2.0]))
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ view[:3, :].T @ intrinsics.T
    if np.any(projected[:, 2] <= 0):
        raise ValueError("SDK view puts forward camera samples behind the image")
    projected = projected[:, :2] / projected[:, 2, None]
    error = float(np.max(np.abs(projected - pixels)))
    if error > 1e-3:
        raise ValueError("SDK camera axes and pinhole projection disagree")
    return dict(
        configuration=configuration,
        intrinsics=intrinsics.tolist(),
        view_matrix_ros=view.tolist(),
        sample_world_points=points.tolist(),
        expected_pixels=pixels.tolist(),
        sdk_pixels=projected.tolist(),
        max_projection_error_px=error,
        focal_length=float(camera.get_focal_length()),
        horizontal_aperture=float(camera.get_horizontal_aperture()),
        vertical_aperture=float(camera.get_vertical_aperture()),
    )


def capture(env, indices):
    manager = env.camera_manager
    records = []
    for index in indices:
        names, cameras = manager.camera_names[index], manager.cameras[index]
        if not names or len(names) != len(cameras) or len(set(names)) != len(names):
            raise ValueError("Native camera name/sensor coverage mismatch")
        for number, (name, camera) in enumerate(zip(names, cameras)):
            record = inspect_camera(camera)
            helper = array(manager.get_camera_intrinsics(number, index))
            if helper.shape != (3, 3):
                raise ValueError("Invalid manager intrinsic matrix")
            record.update(
                env_idx=index,
                camera=name,
                manager_intrinsics=helper.tolist(),
                manager_sdk_max_difference=float(
                    np.max(np.abs(helper - array(record["intrinsics"])))
                ),
            )
            records.append(record)
    return dict(
        schema="physicalrsi.robodojo.camera-diagnostics/v1",
        cameras=records,
        scope="reset-time SDK sensor pose/intrinsics and analytic projection consistency",
        renderer_qualification=False,
        physical_qualification=False,
    )
