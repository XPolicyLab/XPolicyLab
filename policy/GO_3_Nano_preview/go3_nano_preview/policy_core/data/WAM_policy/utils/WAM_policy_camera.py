from typing import Dict, List, Optional, Sequence, Tuple


CAMERA_ALIAS_MAP = {'head': 'head', 'top_head': 'head', 'top': 'head', 'observation.images.top_head': 'head', 'observation.images.top': 'head', 'cam_high': 'head', 'observation.images.cam_high': 'head', 'hand_left': 'hand_left', 'left_hand': 'hand_left', 'left_arm': 'hand_left', 'left': 'hand_left', 'observation.images.hand_left': 'hand_left', 'observation.images.left': 'hand_left', 'cam_left_wrist': 'hand_left', 'observation.images.cam_left_wrist': 'hand_left', 'hand_right': 'hand_right', 'right_hand': 'hand_right', 'right_arm': 'hand_right', 'right': 'hand_right', 'observation.images.hand_right': 'hand_right', 'observation.images.right': 'hand_right', 'cam_right_wrist': 'hand_right', 'observation.images.cam_right_wrist': 'hand_right'}


CAMERA_VIEW_ID_MAP = {'head': 0, 'hand_left': 1, 'hand_right': 2}


def canonicalize_camera_alias(cam_name: str) -> str:
    if not isinstance(cam_name, str):
        raise TypeError(f"Camera name must be a string, got {type(cam_name)!r}")
    return CAMERA_ALIAS_MAP.get(cam_name.strip().lower(), cam_name.strip().lower())


def infer_camera_view_ids(cam_names: Sequence[str]) -> List[int]:
    return [
        CAMERA_VIEW_ID_MAP.get(canonicalize_camera_alias(cam_name), position)
        for position, cam_name in enumerate(cam_names)
    ]


def normalize_mode_positions(positions: Sequence[int], n_available: int) -> List[int]:
    if not isinstance(positions, (list, tuple)) or len(positions) == 0:
        raise ValueError("multi-view mode positions must be a non-empty list")

    normalized = []
    for position in positions:
        position = int(position)
        if position < 0 or position >= n_available:
            raise ValueError(
                f"multi-view mode position {position} is out of range for {n_available} configured cameras"
            )
        normalized.append(position)
    return normalized


def resolve_mode_by_positions(
    available_cam_names: Sequence[str],
    positions: Sequence[int],
    available_view_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[str], List[int]]:
    normalized_positions = normalize_mode_positions(positions, len(available_cam_names))
    full_view_ids = (
        list(available_view_ids)
        if available_view_ids is not None
        else infer_camera_view_ids(available_cam_names)
    )
    if len(full_view_ids) != len(available_cam_names):
        raise ValueError(
            f"view_ids length {len(full_view_ids)} does not match cam_use length {len(available_cam_names)}"
        )
    return (
        [available_cam_names[position] for position in normalized_positions],
        [full_view_ids[position] for position in normalized_positions],
    )


def resolve_mode_cam_names(
    available_cam_names: List[str], requested_cams: List[str]
) -> List[str]:
    resolved_by_alias: Dict[str, str] = {}
    for cam_name in available_cam_names:
        alias = canonicalize_camera_alias(cam_name)
        if alias in resolved_by_alias and resolved_by_alias[alias] != cam_name:
            raise ValueError(
                f"Duplicate camera alias {alias!r} found in {available_cam_names}"
            )
        resolved_by_alias[alias] = cam_name

    resolved = []
    missing = []
    for requested_cam in requested_cams:
        alias = canonicalize_camera_alias(requested_cam)
        if alias not in resolved_by_alias:
            missing.append(requested_cam)
            continue
        resolved.append(resolved_by_alias[alias])

    if missing:
        raise ValueError(
            f"Requested cameras {missing} are not available in configured cam_use={available_cam_names}"
        )

    return resolved
