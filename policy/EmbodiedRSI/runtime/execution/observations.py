"""Publish camera previews and the public robot observation without taking actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# The agent must receive the simulator's native camera frame. The previous
# compact JPEG path changed both resolution and image codec, which can erase
# visual cues needed to distinguish objects and empty grasps.


def public_observation(task) -> dict:
    """Export exactly the detached observation exposed by the program primitive."""
    return task.public_observation()


def camera_images(task) -> dict[str, np.ndarray]:
    from XPolicyLab.policy.EmbodiedRSI.runtime.execution.primitives import rgb_images

    frames = rgb_images(public_observation(task))
    if not frames:
        raise RuntimeError("No public RGB cameras; cannot start an agent without observations")
    return frames


def write_camera_images(task, folder: Path, *, workspace: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    # ``workspace`` is bind-mounted into the agent container. Keep the full
    # frames in the run directory instead, where the monitor can still serve
    # them but the agent cannot attach them to a model request.
    raw_folder = workspace.parent / "raw_observations" / folder.name
    raw_folder.mkdir(parents=True, exist_ok=True)
    images = []
    for camera, rgb in camera_images(task).items():
        if not camera.replace("_", "").isalnum():
            raise ValueError("Unsafe camera filename")
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError(f"Invalid RGB camera {camera}: {rgb.shape}/{rgb.dtype}")
        frame = Image.fromarray(rgb)
        raw_path = raw_folder / f"raw_{camera}.png"
        frame.save(raw_path, format="PNG")

        # Keep the established current_<camera>.png name because agent
        # prompts refer to it. It is deliberately the native frame.
        preview_path = folder / f"current_{camera}.png"
        frame.save(preview_path, format="PNG")
        images.append(str(preview_path.relative_to(workspace)))
    return images


def write_public_observation(task, folder: Path) -> None:
    """Serialize the public getter, never env.reset() info or private state."""
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Object arrays are not public observation artifacts")
            key = f"array_{len(arrays):04d}"
            arrays[key] = value
            return {"npz_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [encode(v) for v in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise TypeError(f"Unsupported public observation value: {type(value).__name__}")

    structure = encode(public_observation(task))
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "arrays.npz", **arrays)
    (folder / "observation.json").write_text(json.dumps(structure, indent=2) + "\n")


def export_observation(task, folder: Path, *, workspace: Path) -> list[str]:
    images = write_camera_images(task, folder, workspace=workspace)
    write_public_observation(task, folder)
    manifest = folder / "images.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "images": images,
                # The monitor may add run-local raw references, but the agent
                # receives the same native-resolution PNG listed here.
                "raw_images": [],
                "source": "native-resolution PNG camera frames",
            },
            indent=2,
        )
    )
    temporary.replace(manifest)
    return images
