"""B20 explicit bounded guard for a geometry-qualified open retreat.

Independent state-machine repair; no new perception threshold or object truth.
"""
def guard_axial_capture(policy):
    pending = policy.pending_bottle
    if (pending is None or not pending.get("axial_retreat") or
            pending.get("capture_applied") or policy.index >= len(policy.plan)):
        return
    stage = policy.plan[policy.index]
    if stage.name != "bottle_insert_fingers" or stage.gripper != 1.:
        return
    # The parent increments dwell after its loose waypoint arrival check.
    # Reset it each observation until the separate capture-volume check passes.
    # Do not encode a permission condition as a long, eventually expiring dwell.
    policy.dwell = 0
    if policy.age >= 40:
        policy.age = 160  # parent's next step enters existing open-only recovery
    policy.program_notes["b20_capture_guard"] = dict(
        waiting=True, deadline_steps=40, timed_out=policy.age >= 160)
