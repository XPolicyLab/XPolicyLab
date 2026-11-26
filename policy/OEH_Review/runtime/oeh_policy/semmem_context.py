"""Semantic planning and its frozen STMem evidence; no execution-state projection or model calls."""
from .contracts import plain
from .task_tree import policy_name


def evidence_slot(runtime, packet):
    """One supplied STMem bundle per observation/time, not a new measurement or search."""
    worker = getattr(runtime, "stmem", None)
    spatial = getattr(worker, "last_context", None) or packet.get("spatial_context") or {}
    scene = packet.get("scene_frame") or {}
    refs = [packet["observation_id"]] if packet.get("observation_id") else []
    if scene.get("frame_id"):
        refs.append(scene["frame_id"])
    for entity in spatial.get("entities", []):
        ref = entity.get("evidence")
        if isinstance(ref, str) and ref:
            refs.append(ref)
    refs = list(dict.fromkeys(refs))
    tree = getattr(runtime, "task_tree", None)
    timestamp = spatial.get("snapshot_time", packet.get("sim_time_s"))
    last = tree.state["last_departure_time"] if tree else None
    return {"source": "stmem_context_delivery", "provided_at": timestamp,
            "measurement_time": spatial.get("snapshot_time"), "observation_id": packet.get("observation_id"),
            "scene_frame_id": scene.get("frame_id"), "status": spatial.get("status", "unknown"),
            "available_refs": refs, "owner_node_id": tree.current_id if tree else None,
            "usable_for_departure": timestamp is not None and (last is None or timestamp > last)}


def model_context(runtime, packet, *, full=False, include_partial=True):
    context = {"policy": policy_name(runtime.cfg.get("semmem")), "episode_id": runtime.episode_id,
               "evidence": evidence_slot(runtime, packet)}
    tree = getattr(runtime, "task_tree", None)
    if tree:
        context.update(tree.context(full=full, include_partial=include_partial))
    else:
        context.update(schema="oeh.semmem.base.v2", current_node=packet.get("instruction"), current_node_id=None)
    return plain(context)
