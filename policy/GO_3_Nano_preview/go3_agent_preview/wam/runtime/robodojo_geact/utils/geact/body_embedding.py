from typing import Mapping, Optional


DEFAULT_BODY_EMBEDDING_MAPPING = {
    "g1_real": 0,
    "g2_real": 1,
    "g2_90D": 2,
    "g2_sim": 3,
    "sl_agilex": 4,
    "sl_acone": 5,
    "sl_arxvr": 6,
    "libero": 7,
    "droid": 8,
    "ego_hand": 9,
    "ego_gripper": 10,
    "other": 11,
    "robodojo": 12,
    "robodojo_real": 13,
}

DEFAULT_NUM_BODY_EMBEDDINGS = max(DEFAULT_BODY_EMBEDDING_MAPPING.values()) + 1

BODY_EMB_IDS_JOINT_DIM_7 = tuple(
    DEFAULT_BODY_EMBEDDING_MAPPING[name]
    for name in ("g1_real", "g2_real", "g2_90D", "g2_sim", "droid")
)
BODY_EMB_IDS_JOINT_DIM_6 = tuple(
    DEFAULT_BODY_EMBEDDING_MAPPING[name]
    for name in (
        "sl_agilex",
        "sl_acone",
        "sl_arxvr",
        "libero",
        "ego_hand",
        "ego_gripper",
        "other",
        "robodojo",
        "robodojo_real",
    )
)
BODY_EMB_IDS_BY_JOINT_DIM = {
    7: BODY_EMB_IDS_JOINT_DIM_7,
    6: BODY_EMB_IDS_JOINT_DIM_6,
}


def _normalized(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def infer_body_type_from_dataset_name(dataset_name: object) -> Optional[str]:
    name = _normalized(dataset_name)
    if 'ego' in name and 'hand' in name:
        return 'ego_hand'
    if 'ego' in name and 'gripper' in name:
        return 'ego_gripper'
    if "90d" in name:
        return "g2_90D"
    if "sim" in name:
        return "g2_sim"
    if "g1_real" in name:
        return "g1_real"
    if "g2_real" in name:
        return "g2_real"
    if "agilex" in name:
        return "sl_agilex"
    if "arxvr" in name or "arx_vr" in name:
        return "sl_arxvr"
    if "acone" in name or "sl_arx" in name:
        return "sl_acone"
    if "libero" in name:
        return "libero"
    if "droid" in name:
        return "droid"
    if "robodojo_real" in name:
        return "robodojo_real"
    if "robodojo" in name:
        return "robodojo"
    return 'other'


def resolve_body_emb_id(
    body_type: object,
    mapping: Optional[Mapping[object, int]] = None,
) -> int:
    table = dict(DEFAULT_BODY_EMBEDDING_MAPPING if mapping is None else mapping)
    normalized_table = {_normalized(name): int(idx) for name, idx in table.items()}
    key = _normalized(body_type)
    if key not in normalized_table:
        raise KeyError(
            f"Unknown body_type={body_type!r}. Expected one of "
            f"{sorted(normalized_table)}. Set data.*.body_type_list explicitly."
        )
    body_id = normalized_table[key]
    if body_id < 0:
        raise ValueError(f"body embedding ID must be non-negative, got {body_id} for {body_type!r}")
    return body_id


def body_type_for_emb_id(
    body_emb_id: int,
    mapping: Optional[Mapping[object, int]] = None,
) -> str:
    table = dict(DEFAULT_BODY_EMBEDDING_MAPPING if mapping is None else mapping)
    reverse = {int(idx): _normalized(name) for name, idx in table.items()}
    key = int(body_emb_id)
    if key not in reverse:
        raise KeyError(
            f"body_emb_id={body_emb_id} has no entry in the body embedding table "
            f"{sorted(table)}. Extend the mapping to cover this embodiment."
        )
    return reverse[key]


def validate_body_embedding_mapping(mapping: Mapping[object, int], num_embeddings: int) -> dict:
    normalized = {_normalized(name): int(idx) for name, idx in mapping.items()}
    if not normalized:
        raise ValueError("body_embedding.mapping must not be empty")
    ids = list(normalized.values())
    if len(set(ids)) != len(ids):
        raise ValueError(f"body_embedding.mapping contains duplicate IDs: {normalized}")
    if min(ids) < 0 or max(ids) >= int(num_embeddings):
        raise ValueError(
            f"body_embedding.mapping IDs must be in [0, {int(num_embeddings) - 1}], got {normalized}"
        )
    return normalized
