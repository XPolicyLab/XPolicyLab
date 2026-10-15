"""Make ``annotator.get_data()`` return the frame for the current physics state.

``get_data()`` hands back whatever the SDG graph last wrote; it never waits.
How many ``env.render()`` calls it takes for a physics change to reach that
buffer depends on Kit's update order:

* With ``/app/updateOrder/checkForHydraRenderComplete=1000`` plus
  ``/app/renderer/waitIdle`` and ``/app/hydraEngine/waitIdle`` (the settings
  Isaac Lab ships in its ``*.rendering.kit`` experiences under "Avoids frame
  offset issue"), an app update blocks until its own frame is rendered and
  dispatched. One render after the physics step is then exactly current,
  regardless of how many render products exist.
* Without them the frame produced by an update reflects the state of the
  previous update, and under load frames are dropped or queued, so the buffer
  can be several frames old. A fixed render count cannot fix that reliably.

This project launches with ``isaaclab.python.kit``, which lacks those settings,
so the entrypoints append them through :func:`add_zero_delay_kit_args`.

:func:`wait_for_latest_cameras` renders until the SDG ``PostProcessDispatcher``
has delivered the required number of new frames. It counts deliveries by
watching the dispatcher's reference time change, which is the same clock
``rep.orchestrator.step(wait_for_render=True)`` polls; it is *not* comparable
with PhysX ``current_time``, so no absolute comparison is made. ``env.render()``
does not advance physics.
"""

from __future__ import annotations

from typing import Optional

_DISPATCHER_PATH = "/Render/PostProcess/SDGPipeline/PostProcessDispatcher"

# Kit settings that make each app update wait for its own rendered frame.
ZERO_DELAY_KIT_SETTINGS = {
    "/app/updateOrder/checkForHydraRenderComplete": "1000",
    "/app/renderer/waitIdle": "true",
    "/app/hydraEngine/waitIdle": "true",
}

_warned_missing_settings = False
_reported_active = False


def zero_delay_kit_args() -> str:
    """Kit command-line overrides for the zero-delay render settings."""
    return " ".join(f"--{key}={value}" for key, value in ZERO_DELAY_KIT_SETTINGS.items())


def add_zero_delay_kit_args(args) -> None:
    """Append the zero-delay overrides to an ``AppLauncher`` argparse namespace.

    Must run before ``AppLauncher(args)``; the update-order setting is read at
    startup and cannot be changed afterwards.
    """
    current = getattr(args, "kit_args", None) or ""
    missing = [
        f"--{key}={value}"
        for key, value in ZERO_DELAY_KIT_SETTINGS.items()
        if f"--{key}=" not in current
    ]
    if missing:
        args.kit_args = (current + " " + " ".join(missing)).strip()


def zero_delay_active() -> bool:
    """True when the running Kit app has the zero-delay settings applied."""
    try:
        import carb.settings

        settings = carb.settings.get_settings()
    except Exception:
        return False
    order = settings.get("/app/updateOrder/checkForHydraRenderComplete")
    try:
        if order is None or int(order) < 1000:
            return False
    except (TypeError, ValueError):
        return False
    return _truthy(settings.get("/app/renderer/waitIdle")) and _truthy(
        settings.get("/app/hydraEngine/waitIdle")
    )


def _truthy(value) -> bool:
    """Command-line overrides may arrive as strings; ``bool("false")`` is True."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def dispatcher_time() -> Optional[float]:
    """Reference time of the last frame the SDG pipeline dispatched."""
    try:
        import omni.graph.core as og

        node = og.get_node_by_path(_DISPATCHER_PATH)
        if node is None:
            return None
        den = node.get_attribute("outputs:referenceTimeDenominator").get()
        if not den:
            return None
        return float(node.get_attribute("outputs:referenceTimeNumerator").get()) / float(den)
    except Exception:
        return None


def _warn_missing_settings() -> None:
    global _warned_missing_settings
    if _warned_missing_settings:
        return
    _warned_missing_settings = True
    print(
        "[render_sync] zero-delay Kit settings are not active; camera frames "
        "lag physics by at least one render. Launch with: " + zero_delay_kit_args()
    )


def _report_active() -> None:
    global _reported_active
    if _reported_active:
        return
    _reported_active = True
    print("[render_sync] zero-delay Kit settings active; one delivered frame per capture.")


def wait_for_latest_cameras(
    env,
    capture_manager=None,
    min_passes: int = 1,
    max_passes: int = 16,
) -> int:
    """Render until the dispatcher has delivered ``min_passes`` new frames.

    With the zero-delay settings one delivered frame carries the current
    physics state. Without them the pipeline is one frame deep, so one more
    delivery is required. Returns the number of ``env.render()`` calls made.
    """
    del capture_manager  # kept for call compatibility; the fence is global
    required = max(1, int(min_passes))
    if not zero_delay_active():
        _warn_missing_settings()
        required += 1
    else:
        _report_active()
    max_passes = max(required, int(max_passes))

    last_stamp = dispatcher_time()
    delivered = 0
    passes = 0
    while passes < max_passes:
        env.render()
        passes += 1
        stamp = dispatcher_time()
        if stamp is None or last_stamp is None:
            # No dispatcher clock to watch: fall back to counting renders.
            delivered = passes
        elif stamp != last_stamp:
            delivered += 1
        last_stamp = stamp
        if delivered >= required:
            break
    return passes
