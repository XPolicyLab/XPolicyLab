"""Public execution state, independent of semantic planning and memory policies."""
import copy

from .contracts import plain, STATE

ROUTING_FIELDS = {"request_id", "candidate_id", "observation_id", "decision", "native_step"}
MEMORY_FIELDS = {"spatial_context", "semantic_context", "state_context", "images", "scene_frame"}


def model_context(runtime, packet):
    """Use the existing decision projection once, retaining other public framework fields."""
    framework = {k: copy.deepcopy(v) for k, v in packet.items() if k not in MEMORY_FIELDS}
    spec = packet.get("task_spec")
    framework.setdefault("task_spec", None)
    cfg = runtime.cfg
    if cfg.get("review", {}).get("teacher", {}).get("input_context"):
        from .input_context import packet as project_packet, rounded
        projected = project_packet(framework, runtime.method.limits["student_max_steps"])
        framework = rounded(framework)
        framework.update(projected)
    state = getattr(runtime, "observation", None)
    result = {"schema": "oeh.state-context.v1", "episode_id": runtime.episode_id,
        "framework_context": framework,
        "constraints": {"task_spec_source": {k: copy.deepcopy(spec[k]) for k in ("id", "source_pin") if k in spec} if isinstance(spec, dict) else None,
            "scope": "episode; current observation for machine state", "preconditions": packet.get("preconditions"),
            "unresolved_assumptions": packet.get("unresolved_assumptions", (spec or {}).get("role_assumption")),
            "action_limits": copy.deepcopy(cfg.get("action_limits")),
            "mode_limits": copy.deepcopy(getattr(runtime.method, "limits", None)),
            "token_allowance": copy.deepcopy(cfg.get("review", {}).get("llm_token_budget")),
            "final_home_steps": getattr(runtime, "final_home_steps", None)},
        "machine_state": {k: copy.deepcopy(v) for k, v in state.get("state", {}).items() if k in STATE} if isinstance(state, dict) else None}
    return plain(result)


def observation_packet(packet):
    """Routing and image references only; execution state is delivered separately."""
    return {k: copy.deepcopy(v) for k, v in packet.items() if k in ROUTING_FIELDS or k in {"images", "scene_frame"}}
