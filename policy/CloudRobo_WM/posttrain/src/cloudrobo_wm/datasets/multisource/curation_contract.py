import hashlib
import importlib.metadata
from cloudrobo_wm.storage import DataPath as Path
import numpy as np
from .schema import fingerprint, file_sha256
VERSION = 'multisource.curation.v2'
DECISIONS = dict(nonfinite='trim', quaternion_invalid='trim', clock='trim', invalid_language='trim', spike='review', extreme='review', blank='review', blur='review', exposure='review', flicker='review', jump='review', frozen='review', frozen_view_during_motion='review', decode_error='drop', pts='drop')

def v2_policy():
    from .curation import DEFAULT_POLICY
    return dict(DEFAULT_POLICY, version=VERSION, decisions=dict(DECISIONS), calibration_seed=42, calibration_folds=2)

def validate_v2(policy):
    expected = v2_policy()
    if set(policy) != set(expected) or set(policy['decisions']) != set(DECISIONS):
        raise ValueError('Incomplete v2 quality policy')
    if set(policy['decisions'].values()) - {'review', 'trim', 'drop'}:
        raise ValueError('Quality decisions must be review/trim/drop')
    if any((policy['decisions'][x] != 'drop' for x in ('decode_error', 'pts'))):
        raise ValueError('Unreadable or misindexed media cannot enter training')
    if policy['calibration_folds'] != 2 or type(policy['calibration_seed']) is not int:
        raise ValueError('Expected deterministic two-fold calibration')

def fold(identity, seed=42):
    return int.from_bytes(hashlib.sha256(f'fold:{seed}:{identity}'.encode()).digest()[:8], 'big') % 2

def stratified_ids(episodes, source_id, limit, seed=42):
    if limit < 2:
        raise ValueError('Cross-fitting needs at least two reference episodes per task')
    result = []
    for f in range(2):
        candidates = [ep for ep in episodes if fold(f"{source_id}:{ep['episode_index']}", seed) == f]
        budget = (limit + 1 - f) // 2
        for chunk in np.array_split(np.asarray(candidates, dtype=object), min(budget, len(candidates)) or 1):
            if len(chunk):
                result.append(min(chunk, key=lambda ep: hashlib.sha256(f"sample:{seed}:{source_id}:{ep['episode_index']}".encode()).digest()))
    return sorted(result, key=lambda ep: ep['episode_index'])

def decide(length, evidence, policy):
    hard = np.zeros(length, bool)
    drop = []
    for kind, intervals in evidence.items():
        action = policy['decisions'][kind]
        if intervals and action == 'drop':
            drop.append(kind)
        if action == 'trim':
            for a, b in intervals:
                hard[a:b] = True
    return (hard, drop)

def implementation_contract(*, video=False):
    root = Path(__file__).resolve().parents[4]
    modules = ['scripts/curate_agibot.py', 'scripts/clean_agibot_idle.py', 'scripts/agibot_media_common.py']
    modules += [f'src/cloudrobo_wm/datasets/multisource/{name}.py' for name in ('curation', 'curation_contract', 'agibot_cleaning', 'agibot_sources', 'curation_report', 'schema')]
    if video:
        modules += ['src/cloudrobo_wm/datasets/multisource/agibot_video.py']
    versions = {}
    for package in ('numpy', 'scipy', 'pyarrow', *(('av', 'opencv-python') if video else ())):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    if video:
        import cv2, av
        versions['opencv_runtime'] = cv2.__version__
        versions['ffmpeg_libraries'] = {k: list(v) for k, v in av.library_versions.items()}
    values = dict(files={name: file_sha256(root / name) for name in modules}, packages=versions)
    return dict(**values, sha256=fingerprint(values))

def training_implementation():
    root = Path(__file__).resolve().parents[4]
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += [root / 'scripts' / name for name in ('multisource_prepare.py', 'audit_multisource_labels.py', 'prepare_agibot_sources.py', 'verify_agibot_training_bundle.py', 'run_agibot_finalize.py', 'curate_agibot.py', 'clean_agibot_idle.py', 'agibot_media_common.py', 'audit_agibot_curation.py')]
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}
