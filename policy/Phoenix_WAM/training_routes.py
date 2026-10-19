"""Accept only explicitly reviewed direct-resize training data contracts."""

from omegaconf import OmegaConf

from .joint96_assets import read_release


ONLINE = 'openwam_raw_hdf5_joint14_direct_resize256'
TRIMMED = 'openwam_raw_hdf5_joint14_direct_resize256_static_trim_v1'
CACHED = 'openwam_h5_joint14_static_trim_bicubic_paired_cache_v1'
ROUTES = {
    ONLINE: ('wam.datasets.robodojo_h5_resize256.RoboDojoH5Resize256Dataset', False),
    TRIMMED: ('wam.datasets.robodojo_h5_static_trim.RoboDojoH5StaticTrimDataset', False),
    CACHED: ('wam.datasets.robodojo_h5_trim_cache.RoboDojoH5TrimCachedDataset', True),
}
TRIM_VERSION = 'conservative_pause_full_frame_audit_v2'
TRIM_SHA = '26cbd91ee15c0c3b998b09ce15f0fdb0fff2d194f7876fce6e12ccb3f4898757'
CSV_SHA = '1d3b3afd54388dc5283557f90694116b79f86a4f3809017bf9e7a6cd386e4852'
FAMILY_COUNTS = {
    'arx_x5': (540, 44, 525, 589210),
    'piper': (600, 437, 6830, 532307),
    'piper_x': (600, 124, 1556, 404877),
}


def validate_route(cfg, contract, family):
    """Validate cache/trim provenance without needing training data at deployment."""
    dataloader = contract.get('dataloader')
    if dataloader not in ROUTES:
        raise ValueError('Require a reviewed direct-resize checkpoint, not old crop/cache weights')
    reader, cached = ROUTES[dataloader]
    if contract.get('cached_visuals') is not cached:
        raise ValueError('Checkpoint visual-cache flag disagrees with its training route')
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    release = read_release()
    if dataloader == ONLINE:
        if contract.get('temporal_trim') is not None or get('joint96_posttrain.temporal_trim') is not None:
            raise ValueError('Untrimmed route carries an unexpected trim contract')
    else:
        episodes, cuts, removed, windows = FAMILY_COUNTS[family]
        expected = dict(
            version=TRIM_VERSION,
            manifest_sha256=TRIM_SHA,
            source_csv_sha256=CSV_SHA,
            source_raw_index_sha256=contract.get('raw_index_sha256'),
            family=family,
            episodes=episodes,
            cuts=cuts,
            removed_frames=removed,
            windows=windows,
            sample_start_rule='retained_length_minus_one',
            action_target='absolute_joint14_next_retained_state',
            visual_source='same_retained_frame_mapping_all_three_h5_cameras',
            normalization_statistics='unchanged_original_family_q01_q99',
            original_h5_modified=False,
        )
        if contract.get('temporal_trim') != expected:
            raise ValueError('Unreviewed temporal-trim contract or family counts')
        sources = release.get('trim_training_source_sha256')
        if not sources or contract.get('temporal_trim_source_sha256') != sources:
            raise ValueError('Unreviewed temporal-trim training source identity')
        if contract.get('action_target_validity') != 'next_retained_state_exists_exclude_terminal_v1':
            raise ValueError('Unexpected terminal action-mask semantics')
        if get('joint96_posttrain.temporal_trim') != TRIM_VERSION or get('data.train.trim_manifest_sha256') != TRIM_SHA:
            raise ValueError('Saved config has a different temporal-trim policy')

    if cached:
        expected_sha = release.get('trim_cache_contract_sha256')
        if not expected_sha or contract.get('cache_contract_sha256') != expected_sha:
            raise ValueError('Unreviewed paired visual cache identity')
        required = {
            'visual_source': 'reviewed_retained_h5_timeline_clean_bicubic',
            'image_color_jitter': False,
            'window_coverage': 'retained_length_minus_one',
            'windows': FAMILY_COUNTS[family][3],
        }
        for key, value in required.items():
            if contract.get(key) != value:
                raise ValueError('Cached training contract mismatch: ' + key)
        if not get('data.train.visual_cache.root'):
            raise ValueError('Saved config is missing the paired-cache binding')
        for name in ('image_color_jitter', 'image_camera_jitter', 'image_sensor_noise', 'image_domain_jitter'):
            if get(f'data.train.{name}.enabled', False):
                raise ValueError('Clean visual cache cannot have online image augmentation')
    elif get('data.train.visual_cache') is not None:
        raise ValueError('Online training must not reference a visual cache')

    return {'dataloader': dataloader, 'reader': reader, 'cached': cached}
