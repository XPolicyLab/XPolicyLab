"""Optional startup ordering for heavily concurrent, headless Kit processes."""

import asyncio
import time


def configure_collision_stack(physx, requested=0):
    """Increase a host-selected allocation without changing solver parameters."""
    original = physx.gpu_collision_stack_size
    if type(requested) is not int or requested < 0 or (requested and requested < original):
        raise ValueError("Collision stack capacity must be zero or an integer >= the native default")
    physx.gpu_collision_stack_size = requested or original
    return {"original_bytes": original, "requested_bytes": requested,
            "effective_bytes": physx.gpu_collision_stack_size}


def configure_rigid_patch_count(physx, requested=0):
    """Increase contact patch storage without changing dynamics or the solver."""
    original = physx.gpu_max_rigid_patch_count
    if type(requested) is not int or requested < 0 or (requested and requested < original):
        raise ValueError("Rigid patch capacity must be zero or an integer >= the native default")
    physx.gpu_max_rigid_patch_count = requested or original
    return {"original_count": original, "requested_count": requested,
            "effective_count": physx.gpu_max_rigid_patch_count}


def wait_for_renderer(app, timeout=1200):
    """Finish two rendered frames before CuRobo may start CUDA graph capture.

    SimulationApp's short viewport wait can time out and continue while OptiX
    is still initializing. Drive Kit until its render-completion API confirms
    readiness, without creating a physics world or changing rendering settings.
    """
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Renderer startup barrier requires an active viewport")
    return _wait_for_frames(app, viewport, timeout)


def _wait_for_frames(app, viewport, timeout):
    started = time.monotonic()
    previous = viewport.updates_enabled
    viewport.updates_enabled = True
    future = asyncio.ensure_future(viewport.wait_for_rendered_frames(2))
    try:
        while not future.done():
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Renderer did not deliver startup frames within {timeout}s")
            app.update()
            time.sleep(0.01)
        future.result()  # Surface asynchronous renderer failures before planner construction.
        return time.monotonic() - started
    finally:
        if not future.done():
            future.cancel()
        viewport.updates_enabled = previous
