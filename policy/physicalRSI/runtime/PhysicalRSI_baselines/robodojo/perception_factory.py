"""Version 2 perception factory with caller-owned RGB workers and leases."""

from pathlib import Path

import numpy as np

from PhysicalRSI_core.infra.storage import file_digest, identifier

from .camera_geometry import PinholeCamera
from .depth_alignment import PlaneDepthAlignment
from .grounded_scene import GroundedScene
from .rgb_service import RGBService, kind, verify


def create(*, configuration, own, identity, pool=None):
    config = configuration
    camera = PinholeCamera(**config["camera"])
    if camera.width * camera.height > 1024 * 1024:
        raise ValueError("Bounded perception camera required")
    mask_file = Path(config["plane_mask"]["path"]).resolve(strict=True)
    if file_digest(mask_file) != config["plane_mask"]["sha256"]:
        raise ValueError("Public plane mask changed")
    mask = np.load(mask_file, allow_pickle=False)
    if mask.dtype != bool or mask.shape != (camera.height, camera.width):
        raise ValueError("Plane mask differs from calibrated camera")
    alignment = PlaneDepthAlignment(camera=camera, **config["alignment"])
    if set(config["services"]) != {"targets", "depth"}:
        raise ValueError("Both perception services required")
    for name, entry in config["services"].items():
        verify(entry["spec"])
        if kind(entry["spec"]) != name or entry["spec"]["image_shape"] != [
            camera.height,
            camera.width,
            3,
        ]:
            raise ValueError("Perception service kind/camera mismatch")
    root = (
        Path(config["output"])
        / identifier(identity["episode"])
        / identifier(identity["actor"])
        / "perception"
    )
    services = {}
    for name in ("targets", "depth"):
        entry = config["services"][name]
        # Register before startup so partial failures remain reachable by the
        # existing rollback/cleanup, including failures to kill a child process.
        service = own(
            RGBService(
                entry["spec"],
                python=entry["python"],
                environment=entry["environment"],
                output=root / name,
                pool=pool,
                resources=entry["resources"],
                startup_timeout_s=entry["startup_timeout_s"],
            )
        )
        service.__enter__()
        services[name] = service
    return GroundedScene(
        camera_name=config["camera_name"],
        camera=camera,
        depth_service=services["depth"],
        target_service=services["targets"],
        alignment=alignment,
        plane_mask=mask,
        geometry=config["geometry"],
    )
