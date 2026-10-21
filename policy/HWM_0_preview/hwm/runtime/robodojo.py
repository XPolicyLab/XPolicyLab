"""Optional RoboDojo capture path without intermediate video writes."""


def check_intermediate_safety(task_env, indices):
    """Retain observation-side safety checks even when no image is read."""
    if task_env.physx_monitor_enabled:
        task_env._check_physx_broken_envs()
        task_env._check_endpose_finite(indices)


def history_head_observations(
    task_env, indices, *, head_camera: str, check_safety: bool = True
):
    """Use official capture, retaining render sync and safety checks.

    Only intermediate action-chunk steps use this path. Full observations at
    action boundaries still use the benchmark's get_obs[_batch] API.
    """
    if check_safety:
        check_intermediate_safety(task_env, indices)
    manager = task_env.obs_manager
    camera_ids = [
        list(manager.camera_manager.camera_names[index]).index(head_camera)
        for index in indices
    ]
    if len(set(camera_ids)) != 1:
        raise ValueError("RoboDojo head camera slots differ across environments")
    manager.render_for_capture()
    captured = manager.capture_manager.step(env_ids=indices, cam_ids=[camera_ids[0]])
    frames = captured[0]["rgb"]
    if len(frames) != len(indices):
        raise ValueError("RoboDojo head capture/environment count mismatch")
    return [
        {
            "env_idx": index,
            "instruction": manager.instruction[index],
            "vision": {head_camera: {"color": frame["data"][:, :, :3]}},
        }
        for index, frame in zip(indices, frames, strict=True)
    ]
