"""Robotwin conditioning IDs; retain the pretrained tensor shape."""
from typing import Mapping, Optional

DEFAULT_BODY_EMBEDDING_MAPPING = {"robotwin": 11}
# Unused rows remain reserved so existing 14-row checkpoints load unchanged.
DEFAULT_NUM_BODY_EMBEDDINGS = 14
BODY_EMB_IDS_BY_JOINT_DIM = {6: (11,)}

def _normalized(value: object) -> str:
    return str(value).strip().lower()

def validate_body_embedding_mapping(mapping: Mapping[object, int], num_embeddings: int) -> dict:
    normalized = {_normalized(name): int(idx) for name, idx in mapping.items()}
    if normalized != DEFAULT_BODY_EMBEDDING_MAPPING:
        raise ValueError('Only robotwin with body embedding ID 11 is supported')
    if int(num_embeddings) != DEFAULT_NUM_BODY_EMBEDDINGS:
        raise ValueError('The checkpoint-compatible body embedding table requires 14 rows')
    return normalized

def infer_body_type_from_dataset_name(dataset_name: object) -> Optional[str]:
    if _normalized(dataset_name) != 'robotwin':
        raise ValueError('Only the robotwin dataset type is supported')
    return 'robotwin'

def resolve_body_emb_id(body_type: object, mapping: Optional[Mapping[object, int]] = None) -> int:
    table = validate_body_embedding_mapping(DEFAULT_BODY_EMBEDDING_MAPPING if mapping is None else mapping, DEFAULT_NUM_BODY_EMBEDDINGS)
    if _normalized(body_type) != 'robotwin':
        raise ValueError('Only robotwin robot type is supported')
    return table['robotwin']

def body_type_for_emb_id(body_emb_id: int, mapping: Optional[Mapping[object, int]] = None) -> str:
    validate_body_embedding_mapping(DEFAULT_BODY_EMBEDDING_MAPPING if mapping is None else mapping, DEFAULT_NUM_BODY_EMBEDDINGS)
    if int(body_emb_id) != 11:
        raise ValueError('Only robotwin body embedding ID 11 is supported')
    return 'robotwin'
