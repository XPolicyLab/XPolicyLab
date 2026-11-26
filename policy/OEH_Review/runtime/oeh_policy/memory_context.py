"""Bounded model projection; the on-disk object archive remains complete."""
import copy
import json


def spatial_context(snapshot, *, max_entities=12, max_chars=24000, entity_ids=()):
    preferred = set(entity_ids)
    entities = snapshot.get("entities", [])
    ordered = sorted(entities, key=lambda e: (e.get("entity_id") not in preferred,
        e.get("visibility") != "visible", e.get("age_s", 0), e.get("entity_id", "")))
    result = {"schema": "oeh.stmem.compact.v1", "snapshot_time": snapshot.get("snapshot_time"),
              "status": snapshot.get("status", "ok"), "entities": [], "total_entities": len(entities),
              "description_states":copy.deepcopy(snapshot.get("description_states",{})),
              "pending_observations":snapshot.get("pending_observations",0),
              "initial_review_status":snapshot.get("initial_review_status","unknown"),
              "omitted_entities": len(entities), "history": "Query resolve/current/at/trajectory; unseen is not absent."}
    fields = ("entity_id", "label", "camera", "detector_label", "description_status", "identity_status", "lifecycle", "instruction_mentioned", "interactable", "avoid_interaction", "time", "frame_id", "box", "model_dimensions",
              "visibility", "motion", "age_s", "velocity_m_s", "angular_velocity_rad_s",
              "position_uncertainty_m", "orientation_ambiguous", "observed_or_predicted", "evidence")
    for entity in ordered[:max_entities]:
        item = {k: copy.deepcopy(entity[k]) for k in fields if k in entity}
        item["label"] = str(item.get("label", ""))[:100]
        item["description"] = str(entity.get("description", ""))[:180]
        if entity.get("label_validation") is not None:
            item["label_validation"] = {k:copy.deepcopy(entity["label_validation"].get(k)) for k in ("valid","label")}
        result["entities"].append(item)
        if len(json.dumps(result, ensure_ascii=True)) > max_chars:
            result["entities"].pop()
            break
    result["omitted_entities"] = len(entities)-len(result["entities"])
    return result
