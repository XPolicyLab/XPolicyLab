from wam.utils.runtime_contract import semantic_contract
'Unchanged FastWAM cache contracts, without its dataset imports.'
import hashlib
import json
from pathlib import Path

def require(value, message):
    if not value:
        raise ValueError(f'VAE+STVAE cache: {message}')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def pair_contracts(vae_root, stvae_roots):
    vae_root = Path(vae_root).resolve()
    stvae_roots = [Path(p).resolve() for p in stvae_roots]
    require(len(stvae_roots) == 2 and [p.name for p in stvae_roots] == ['head', 'wrist'] and (stvae_roots[0].parent == stvae_roots[1].parent), 'STVAE head/wrist routing')
    roots = [vae_root / 'head', vae_root / 'wrist', *stvae_roots]
    contracts = [read_json(p / 'contract.json') for p in roots]
    vr = read_json(vae_root / 'contract.json')
    for view, contract in zip(('head', 'wrist'), contracts[:2]):
        require(semantic_contract(contract) == semantic_contract({**vr, 'view': view}), 'VAE view contract identity')
    require(vr['scope'] == 'full' and vr['selection'] is None and (vr['feature_shape_per_view'] == [48, 3, 16, 16]) and (vr['encoder_type'] == 'wan22_vae_per_view'), 'VAE256 native full cache required')
    sr = contracts[2]
    require(semantic_contract({k: v for k, v in sr.items() if k != 'view'}) == semantic_contract({k: v for k, v in contracts[3].items() if k != 'view'}), 'STVAE view contract identity')
    require(all((c['dataset_length'] == 277713 for c in contracts)), 'full LIBERO length')
    for key in ('state_frames', 'action_horizon', 'rgb_offsets', 'latent_frames', 'pad_policy'):
        require(vr['window'][key] == sr['window'][key], f'window mismatch: {key}')
    require(sr['selection'] == 'full_dataset' and sr['posterior'] == 'mean' and (sr['extra_latent_scaling'] == 'none'), 'STVAE deterministic full mean cache required')
    return (roots, contracts)
