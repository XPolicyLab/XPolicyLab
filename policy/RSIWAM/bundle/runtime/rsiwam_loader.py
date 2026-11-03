"""Portable, hash-bound RSIWAM composition. No GPU or network use on import."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'recoverwam_success133_original_parent20261032_lora20261033'
BASE = 'ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba'
PARENT = '861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3'
ADAPTER = 'eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361'
HELPER = '946f25497835c4a18a74731058b7dec25fa1acaf2aad4a4b4f78814cca2a735d'
SOURCE = '95feb351fc1211318d7edec9e3814679835029b7ba4314d04d25464b6a4691f0'
CHECKPOINT = '38b9eaa31383a8a098dc68e568c125d5d22d3bb749742145ac38d7c2a5d3dbb6'
BASE_ASSETS = '955de560be7d371e2df31434c754c8332363d68525856a1da11db56f462c43da'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    path = Path(path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'Missing or changed artifact: {path}')
    return path


def check_package():
    parent = require_hash(ROOT / 'weights/parent/adapter_final.pt', PARENT)
    adapter = require_hash(ROOT / 'weights/step000400/adapter.pt', ADAPTER)
    checkpoint = require_hash(ROOT / 'weights/step000400/checkpoint.json', CHECKPOINT)
    commit = json.loads(checkpoint.read_text())
    identity = commit['identity']
    expected = dict(candidate_id=CANDIDATE, base_identity=BASE,
                    parent_checkpoint_sha256=PARENT, parent_bridge_seed=20261032,
                    recovery_recipe='original', lora_seed=20261033,
                    helper_sha256=HELPER, consumer_source_tree_sha256=SOURCE)
    if commit['step'] != 400 or commit['adapter_sha256'] != ADAPTER or any(
            identity.get(k) != value for k, value in expected.items()):
        raise ValueError('The committed candidate, base, parent, or step differs')
    require_hash(ROOT / 'weights/parent/recipe.json', identity['parent_recipe_sha256'])
    helper = require_hash(ROOT / 'runtime/recovery_action_lora.py', HELPER)
    require_hash(ROOT / 'runtime/rsiwam_native_wrapper.py',
                 '3e99077acecc2a32abd41549a2c3434c8b5b0bc30b49e19a5c0dc8242bd3a00a')
    source = ROOT / 'src/recoverwam'
    hashes = {str(path.relative_to(source)): sha256(path) for path in sorted(source.rglob('*.py'))}
    actual = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != SOURCE:
        raise ValueError('The frozen RecoverWAM source tree differs')
    return dict(parent=parent, adapter=adapter, helper=helper, commit=commit)


def check_base(weights_root):
    """Rehash every external base asset before constructing a model."""
    root = Path(weights_root).expanduser().resolve()
    manifest = json.loads(require_hash(ROOT / 'base_weights.json', BASE_ASSETS).read_text())
    for row in manifest['files']:
        path = root / row['relative_path']
        require_hash(path, row['sha256'])
        if 'bytes' in row and path.stat().st_size != row['bytes']:
            raise ValueError(f'Wrong artifact length: {path}')
    require_hash(root / 'joint2b_s100000/manifest.json', BASE)
    return root


def compose(policy, artifacts=None):
    """Attach frozen parent, then fresh rank-8 Action-LoRA to a loaded base.

    Mirrors reference_switch/switch_runtime.py:load_candidate; this single-model
    entry does not provide the original shared-service admission/controller API.
    """
    artifacts = artifacts or check_package()
    spec = importlib.util.spec_from_file_location('rsiwam_registered_helper', artifacts['helper'])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if Path(helper.lora.__file__).resolve() != (ROOT / 'src/recoverwam/recovery/lora.py').resolve():
        raise ValueError('Another RecoverWAM installation is already imported')
    helper.load_frozen_parent(policy, artifacts['parent'], expected_sha256=PARENT,
                              expected_base_identity=BASE)
    metadata = helper.load_adapter(artifacts['adapter'], policy,
                                   expected_parent_sha256=PARENT, expected_base_identity=BASE)
    if (metadata.get('steps') != 400 or metadata.get('candidate_id') != CANDIDATE
            or metadata.get('training_identity') != artifacts['commit']['identity']):
        raise ValueError('Loaded Action-LoRA metadata differs from committed identity')
    policy.recovery_adapter_metadata = metadata
    policy.model.recovery_mode = 'auto'
    policy.model.requires_grad_(False)
    policy.model.eval()
    policy.model.action_expert.use_gradient_checkpointing = False
    policy.model.mot.mot_checkpoint_mixed_attn = False
    return metadata
