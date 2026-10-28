"""TaskTree's main-policy tools and context delivery; never imports a model client."""
from .task_tree import TaskTree, policy_name


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def update_schema():
    text = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": text}
    task = _object({"subgoal": text, "is_leaf": {"type": ["boolean", "null"]},
        "achievement_condition": {"type": ["string", "null"]}, "required_evidence": strings, "spec_refs": strings})
    children = {"type": "array", "items": task}
    operation = lambda name, fields: _object({"operation": {"type": "string", "enum": [name]}, **fields})
    return _object({"update_id": text, "expected_revision": {"type": "integer", "minimum": 0},
        "operations": {"type": "array", "minItems": 1, "items": {"anyOf": [
            operation("decompose", {"node_id": text, "children": {**children, "minItems": 1}}),
            operation("prepare", {"node_id": text, "achievement_condition": text,
                                  "required_evidence": strings}),
            operation("complete", {"node_id": text, "reason": text, "evidence_refs": {**strings, "maxItems": 1}}),
            operation("replan", {"node_id": text, "levels": {"type": "integer", "minimum": 1},
                "histories": {"type": "array", "minItems": 1, "items": _object({
                    "history": text, "reason": {"type": ["string", "null"]}})}, "children": children})]}}})


def query_schema():
    return _object({"kind": {"type": "string", "enum": ["task_tree_query"]},
        "operation": {"type": "string", "enum": ["subtree", "chain", "chain-span", "children", "full_tree", "all_queues", "queue"]},
        "node_id": {"type": ["string", "null"]}})


def binding_schema():
    return _object({"node_id": {"type": "string", "minLength": 1}, "revision": {"type": "integer", "minimum": 0}})


def ensure_tree(runtime, packet):
    """Root is the supplied instruction, not a task-native automatic decomposition."""
    if policy_name(runtime.cfg.get("semmem")) != "hierarchical-policy":
        return None
    if getattr(runtime, "task_tree", None) is None:
        runtime.task_tree = TaskTree(runtime.output / "semmem", episode_id=runtime.episode_id,
            instruction=packet["instruction"], task_spec=packet.get("task_spec"),
            max_depth=runtime.cfg["semmem"].get("max_depth", 5))
    return runtime.task_tree


def source(packet, call_id, runtime=None, evidence=None):
    result = {"call_id": call_id, "observation_id": packet.get("observation_id"),
            "request_id": packet["request_id"], "sim_time_s": packet["sim_time_s"],
            "native_step": packet.get("native_step")}
    if runtime is not None:
        from .semmem_context import evidence_slot
        evidence = evidence if evidence is not None else evidence_slot(runtime, packet)
        result.update(evidence_time=evidence["provided_at"], available_evidence_refs=evidence["available_refs"],
                      evidence_owner=evidence["owner_node_id"])
    return result


def update(brain, envelope, packet, call_id):
    """A dependent action must not run after an invalid TaskTree mutation."""
    value = envelope.get("semmem_update")
    if value is None:
        return
    tree = getattr(brain.runtime, "task_tree", None)
    replies = getattr(brain, "memory_replies", [])
    try:
        if tree is None:
            raise ValueError("task-tree policy is not enabled")
        result = tree.update(value, source=source(packet, call_id, brain.runtime, getattr(brain, "semmem_evidence", None)))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        replies.append(({"semmem_update_response": {"status": "update_failed", "error": str(exc)}}, []))
        brain.memory_replies = replies
        raise ValueError("TaskTree update failed; dependent action not executed: " + str(exc)) from exc
    replies.append(({"semmem_update_response": result}, []))
    brain.memory_replies = replies
    brain.runtime.emit("semmem_update", {"call_id": call_id, "revision": tree.revision,
                                         "update_id": value["update_id"]})


def query(runtime, value):
    if set(value) != {"kind", "operation", "node_id"}:
        raise ValueError("task_tree_query requires kind, operation and node_id")
    tree = getattr(runtime, "task_tree", None)
    if tree is None:
        raise ValueError("task-tree policy is not enabled")
    return tree.query(value["operation"], value["node_id"])


def validate_binding(runtime, binding, decision):
    tree = getattr(runtime, "task_tree", None)
    if tree is not None and decision["mode"] != "hold":
        tree.check_binding(binding)
    elif binding is not None and tree is None:
        raise ValueError("task_binding requires task-tree policy")
