"""Model presentation of scene metadata. Never changes originals, calibration or image items."""
import copy

IMAGE_FIELDS = ("camera", "frame_id", "native_step", "sim_time", "capture_time", "k", "size", "original_size", "status")


def scene_frame_model_view(frame):
    result = {k: copy.deepcopy(frame[k]) for k in ("frame_id", "native_step", "sim_time", "capture_time") if k in frame}
    images = frame.get("images", [])
    result["images"] = [{k: copy.deepcopy(im[k]) for k in IMAGE_FIELDS if k in im} for im in images]
    refs = [im.get("entity_state_refs", []) for im in images]
    if refs and all(row == refs[0] for row in refs):
        result["entity_state_refs"] = copy.deepcopy(refs[0])
    elif refs:
        result["camera_entity_state_refs"] = {im.get("camera"): copy.deepcopy(im.get("entity_state_refs", [])) for im in images}
    return result


def frames_query_model_view(result):
    # Historical pages can contain multiple times per camera; retain references per image.
    value = {k: v for k, v in result.items() if k not in ("images", "missing_images")}
    for key in ("images", "missing_images"):
        value[key] = [{k: copy.deepcopy(v) for k, v in im.items()
                       if k in IMAGE_FIELDS or k == "entity_state_refs"} for im in result.get(key, [])]
    return value
