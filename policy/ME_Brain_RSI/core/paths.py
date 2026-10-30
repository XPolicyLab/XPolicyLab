"""Resolve configured resource paths relative to the policy directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def policy_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return Path(os.path.abspath(path))


CONFIG_PATH_KEYS = {
    'model_path', 'checkpoint_path', 'tokenizer_path', 'robot_static_path',
    'stack_assets_root', 'nesting_assets_root',
    'equation_assets_root', 'perception_assets_root', 'zyr_assets_root',
    'strategy_evidence_dir', 'focus_vlwa_source', 'policy_uv_env_path',
    'coin_assets_root', 'screws_assets_root',
}
ENVIRONMENT_PATH_KEYS = {
    'CLASSIFY_CLIP_PATH', 'CLASSIFY_DINO_PATH', 'COAP_PUBLIC_RGB_MODELS',
    'FOCUS_VLWA_TOKENIZER_PATH', 'FOCUS_VLWA_SOURCE', 'ME_BRAIN_ROBOT_STATIC',
    'ME_BRAIN_STACK_ASSETS', 'ME_BRAIN_NESTING_ASSETS',
    'ME_BRAIN_EQUATION_ASSETS', 'ME_BRAIN_PERCEPTION_ASSETS', 'ME_BRAIN_ZYR_ASSETS',
    'ME_BRAIN_COIN_ASSETS', 'COIN_TRACE_DIR',
    'ME_BRAIN_SCREWS_ASSETS', 'SCREWS_TRACE_DIR',
    'ME_BRAIN_API_TOKEN_FILE', 'COAP_ASTRA_TOKEN_FILE', 'ME_BRAIN_EVIDENCE_DIR',
}


def normalize_config(config):
    result = dict(config)
    for key in CONFIG_PATH_KEYS:
        if result.get(key):
            result[key] = str(policy_path(result[key]))
    if result.get('source_assets'):
        result['source_assets'] = {name: str(policy_path(path)) for name, path in result['source_assets'].items()}
    return result


def normalize_environment(name, value):
    return str(policy_path(value)) if name in ENVIRONMENT_PATH_KEYS and value else str(value)
