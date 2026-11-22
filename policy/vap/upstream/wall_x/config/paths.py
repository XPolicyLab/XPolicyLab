"""Checkpoint-owned VAP paths are relative to config.yml, never the process cwd."""
from pathlib import Path


# Explicit filesystem fields only: model IDs (including Wan model specs), cache
# identities and dataset names must never be interpreted as local paths.
VAP_PATH_FIELDS = (
    ('checkpoint', 'resume_from'),
    ('checkpoint', 'save_path'),
    ('data', 'dataset_config_path'),
    ('data', 'val_dataset_config_path'),
    ('data', 'normalizer_config', 'customized_action_statistic_dof'),
    ('data', 'dataset_config_effective', 'dataset', 'processors', 'action',
     'params', 'normalizer_config', 'stats_file'),
    ('model', 'text_feature_cache_path'),
    ('model', 'qwen35_model_path'),
)


def resolve_vap_config_paths(raw: dict, config_path) -> dict:
    """Resolve known VAPv2 file fields in place; retain absolute paths and nulls.

    No files are opened here: training-only records may name unavailable assets.
    Call before relocating a config to a per-run checkpoint view, so relative
    references retain their original checkpoint directory as their base.
    """
    if raw.get('model_type') != 'vapv2':
        return raw
    base = Path(config_path).absolute().parent
    for keys in VAP_PATH_FIELDS:
        node = raw
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            continue
        value = node.get(keys[-1])
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        node[keys[-1]] = str(path)
    return raw
