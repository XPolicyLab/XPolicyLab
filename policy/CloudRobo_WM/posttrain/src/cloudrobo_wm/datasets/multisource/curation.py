import numpy as np
VERSION = 'multisource.curation.v1'
DEFAULT_POLICY = dict(version=VERSION, mad_scale=8.0, quantile_alpha=3.0, min_quantile_span=0.02, defect_padding=4, min_interval_frames=5, max_removed_ratio=0.7, video_size=128, video_center_fraction=0.8, frozen_ratio=0.9, frozen_min_seconds=2.0, freeze_run_seconds=1.0, max_pts_bad_ratio=0.02, blur_laplacian=5.0, exposure_fraction=0.98, flicker_mean_delta=35.0, visual_jump_mae=55.0, visual_static_mae=0.1, visual_motion_mae=2.0, protect_effector_events=True, review_filters=[])

def validate_policy(policy):
    if policy.get('version') == 'multisource.curation.v2':
        from .curation_contract import validate_v2
        validate_v2(policy)
        policy = {k: policy[k] for k in DEFAULT_POLICY}
        policy['version'] = VERSION
    if set(policy) != set(DEFAULT_POLICY) or policy['version'] != VERSION:
        raise ValueError('Unknown/incomplete curation policy')
    for key, value in policy.items():
        if key not in ('version', 'review_filters', 'protect_effector_events') and (not np.isfinite(value) or value <= 0):
            raise ValueError('Curation thresholds must be finite and positive')
    for key in ('max_removed_ratio', 'frozen_ratio', 'max_pts_bad_ratio', 'exposure_fraction', 'video_center_fraction'):
        if policy[key] > 1:
            raise ValueError('Invalid curation ratio')
    for key in ('defect_padding', 'min_interval_frames', 'video_size'):
        if type(policy[key]) is not int:
            raise ValueError('Curation frame/size settings must be integers')
    if policy['defect_padding'] < 4:
        raise ValueError('Keep at least four frames away from local defects')
    if not set(policy['review_filters']) <= {'blur', 'exposure', 'flicker', 'jump'}:
        raise ValueError('Unknown visual review filter')
    if type(policy['protect_effector_events']) is not bool:
        raise ValueError('Event protection must be boolean')

def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]

def select_intervals(length, intervals, bad, policy=DEFAULT_POLICY):
    validate_policy(policy)
    valid = np.zeros(length, bool)
    previous = 0
    for a, b in intervals:
        if not 0 <= a < b <= length or a < previous:
            raise ValueError('Invalid original-frame intervals')
        valid[a:b] = True
        previous = b
    bad = np.asarray(bad, bool)
    if bad.shape != (length,):
        raise ValueError('Defect mask length mismatch')
    for a, b in runs(bad):
        valid[max(0, a - policy['defect_padding']):min(length, b + policy['defect_padding'])] = False
    selected = [[a, b] for a, b in runs(valid) if b - a >= policy['min_interval_frames']]
    retained = sum((b - a for a, b in selected))
    if not retained or (length - retained) / length >= policy['max_removed_ratio']:
        return dict(keep=False, intervals=[], retained_frames=0, reason='drop_quality_or_overtrim')
    return dict(keep=True, intervals=selected, retained_frames=retained, reason='keep_curated')

def robust_limit(values, floor, scale):
    median = np.median(values, axis=0)
    return np.maximum(floor, median + scale * 1.4826 * np.median(np.abs(values - median), axis=0))

def fit_quantiles(values, unit, policy=DEFAULT_POLICY):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 100 or (not np.isfinite(x).all()):
        raise ValueError('Quantile calibration needs >=100 finite train samples')
    q1, q99 = np.quantile(x, [0.01, 0.99], axis=0)
    span = np.maximum(q99 - q1, policy['min_quantile_span'])
    return dict(unit=unit, count=len(x), q01=q1.tolist(), q99=q99.tolist(), lower=(q1 - policy['quantile_alpha'] * span).tolist(), upper=(q99 + policy['quantile_alpha'] * span).tolist())

def signal_defects(values, *, kind='scalar', floor=0.01, bounds=None, protect_events=False, policy=DEFAULT_POLICY, evidence=False):
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    x = np.asarray(values, float)
    n = len(x)
    x = x.reshape(n, -1)
    finite = np.isfinite(x).all(axis=1)
    bad = ~finite
    details = {'nonfinite': runs(bad), 'spike': [], 'extreme': []}
    if evidence:
        details['evidence'] = []
        for frame, axis in np.argwhere(~np.isfinite(x)):
            details['evidence'].append(dict(frame=int(frame), axis=int(axis), code='nonfinite', observed=None))
    if kind == 'quaternion':
        norms = np.linalg.norm(x.reshape(n, -1, 4), axis=-1)
        invalid = finite & np.any(np.abs(norms - 1) > 0.1, axis=1)
        details['quaternion_invalid'] = runs(invalid)
        bad |= invalid
        if evidence:
            for frame, axis in np.argwhere(np.abs(norms - 1) > 0.1):
                details['evidence'].append(dict(frame=int(frame), axis=int(axis), code='quaternion_invalid', observed=float(norms[frame, axis]), lower=0.9, upper=1.1))
    if bad.any():
        for a, b in runs(~bad):
            sub_bad, sub_details = signal_defects(x[a:b], kind=kind, floor=floor, bounds=bounds, protect_events=protect_events, policy=policy, evidence=evidence)
            bad[a:b] |= sub_bad
            for key, ranges in sub_details.items():
                if key == 'evidence':
                    details[key].extend((dict(row, frame=row['frame'] + a) for row in ranges))
                    continue
                details.setdefault(key, []).extend([[a + c, a + d] for c, d in ranges])
        return (bad, details)
    if kind == 'quaternion':
        q = x.reshape(n, -1, 4)
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        invalid = np.any(np.abs(norm[..., 0] - 1) > 0.1, axis=1)
        details['quaternion_invalid'] = runs(invalid)
        bad |= invalid
        if invalid.any() or n < 7:
            return (bad, details)
        q = q / norm
        distance = lambda a, b: 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))
        right = q[2:] * np.where(np.sum(q[:-2] * q[2:], axis=-1, keepdims=True) < 0, -1, 1)
        midpoint = q[:-2] + right
        midpoint /= np.maximum(np.linalg.norm(midpoint, axis=-1, keepdims=True), 1e-12)
        residual = np.pad(distance(q[1:-1], midpoint), ((1, 1), (0, 0)))
        step = np.pad(distance(q[1:], q[:-1]), ((1, 0), (0, 0)))
        acceleration = np.abs(np.diff(step, axis=0, prepend=step[:1]))
        jerk = np.abs(np.diff(acceleration, axis=0, prepend=acceleration[:1]))
    else:
        if bounds is not None:
            extreme = ((x < np.asarray(bounds['lower'])) | (x > np.asarray(bounds['upper']))).any(axis=1)
            details['extreme'] = runs(extreme)
            bad |= extreme
            if evidence:
                for frame, axis in np.argwhere((x < bounds['lower']) | (x > bounds['upper'])):
                    details['evidence'].append(dict(frame=int(frame), axis=int(axis), code='extreme', observed=float(x[frame, axis]), lower=bounds['lower'][axis], upper=bounds['upper'][axis]))
        if n < 7:
            return (bad, details)
        window = min(9, n if n % 2 else n - 1)
        trend = savgol_filter(median_filter(x, size=(3, 1), mode='nearest'), window, 2, axis=0, mode='interp')
        residual = np.abs(x - trend)
        acceleration = np.pad(np.abs(np.diff(x, n=2, axis=0)), ((1, 1), (0, 0)))
        jerk = np.pad(np.abs(np.diff(x, n=3, axis=0)), ((1, 2), (0, 0)))
    lim = lambda a, f: robust_limit(a, f, policy['mad_scale'])
    spike_axes = (residual > lim(residual, floor)) & ((acceleration > lim(acceleration, floor)) | (jerk > lim(jerk, floor)))
    candidates = spike_axes.copy()
    if kind != 'quaternion' and protect_events and policy['protect_effector_events']:
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        delta = np.diff(x, axis=0, prepend=x[:1])
        monotone = (minimum_filter1d(delta, 7, axis=0, mode='nearest') >= -floor * 0.1) | (maximum_filter1d(delta, 7, axis=0, mode='nearest') <= floor * 0.1)
        details['protected_effector_events'] = runs((spike_axes & monotone).any(axis=1))
        spike_axes &= ~monotone
    spike = spike_axes.any(axis=1)
    details['spike'] = runs(spike)
    if evidence:
        limits = [lim(value, floor) for value in (residual, acceleration, jerk)]
        for frame, axis in np.argwhere(candidates):
            details['evidence'].append(dict(frame=int(frame), axis=int(axis), code='spike', observed=x[frame].reshape(-1, 4)[axis].tolist() if kind == 'quaternion' else float(x[frame, axis]), residual=float(residual[frame, axis]), residual_limit=float(limits[0][axis]), acceleration=float(acceleration[frame, axis]), acceleration_limit=float(limits[1][axis]), jerk=float(jerk[frame, axis]), jerk_limit=float(limits[2][axis]), protected=bool(not spike_axes[frame, axis]), derivative_basis='per_recorded_frame'))
    return (bad | spike, details)

def trend_alignment(state, command, *, max_lag=8, motion_floor=0.0001):
    from scipy.ndimage import uniform_filter1d
    s, a = (np.asarray(state, float), np.asarray(command, float))
    if s.shape != a.shape or s.ndim != 2 or (not np.isfinite(s).all()) or (not np.isfinite(a).all()):
        return dict(status='unavailable', reason='shape_or_nonfinite')
    if np.array_equal(s, a):
        return dict(status='uninformative', reason='identical_state_command')
    ds, da = [np.diff(uniform_filter1d(x, size=5, axis=0, mode='nearest'), axis=0) for x in (s, a)]
    axes = []
    for i in range(s.shape[1]):
        candidates = []
        for lag in range(-max_lag, max_lag + 1):
            lo, hi = (max(0, -lag), min(len(da), len(ds) - lag))
            u, v = (da[lo:hi, i], ds[lo + lag:hi + lag, i])
            active = (np.abs(u) > motion_floor) & (np.abs(v) > motion_floor)
            if active.sum() < 20 or min(np.std(u), np.std(v)) < motion_floor:
                continue
            corr = float(np.corrcoef(u, v)[0, 1])
            ratio = float(max(np.std(u) / np.std(v), np.std(v) / np.std(u)))
            candidates.append(dict(lag=lag, correlation=corr, amplitude_ratio=ratio, directional_agreement=float(np.mean(u[active] * v[active] > 0))))
        if candidates:
            best = max(candidates, key=lambda r: (r['correlation'], -abs(r['lag'])))
            axes.append(dict(axis=i, **best))
    suspect = any((r['amplitude_ratio'] >= 3 and r['correlation'] < 0.5 or r['directional_agreement'] < 0.6 or r['lag'] < 0 for r in axes))
    return dict(status='review' if suspect else 'measured' if axes else 'uninformative', axes=axes, reason='native_command_semantics_not_certified')

class VideoMetrics:

    def __init__(self, policy=DEFAULT_POLICY):
        self.policy = policy
        self.previous = None
        self.rows = []

    def add(self, rgb):
        import cv2
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB video')
        h, w = image.shape[:2]
        f = self.policy['video_center_fraction']
        dy, dx = (round(h * (1 - f) / 2), round(w * (1 - f) / 2))
        small = cv2.resize(image[dy:h - dy, dx:w - dx], (self.policy['video_size'],) * 2, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        p = self.previous
        self.rows.append([float(gray.mean()), float((gray < 8).mean()), float((gray > 247).mean()), float(np.std(small.reshape(-1, 3).astype(float), axis=0).max()), float(cv2.Laplacian(gray, cv2.CV_32F).var()), float(np.abs(small.astype(float) - p.astype(float)).mean()) if p is not None else 0.0, float(np.array_equal(small, p)) if p is not None else 0.0])
        self.previous = small

    def finish(self, fps):
        x = np.asarray(self.rows, float)
        if not len(x):
            raise ValueError('No decoded video frames')
        p = self.policy
        masks = dict(blank=x[:, 3] < 0.25, blur=x[:, 4] < p['blur_laplacian'], exposure=np.maximum(x[:, 1], x[:, 2]) > p['exposure_fraction'], flicker=np.r_[False, np.abs(np.diff(x[:, 0])) > p['flicker_mean_delta']], jump=x[:, 5] > p['visual_jump_mae'])
        repeat = x[:, 6].astype(bool)
        long_repeat = np.zeros(len(x), bool)
        for a, b in runs(repeat):
            if b - a >= int(np.ceil(fps * p['freeze_run_seconds'])):
                long_repeat[max(0, a - 1):b] = True
        hard = masks['blank'].copy()
        for key in p['review_filters']:
            hard |= masks[key]
        ratio = float(repeat[1:].mean()) if len(x) > 1 else 0.0
        drop = len(x) >= fps * p['frozen_min_seconds'] and ratio >= p['frozen_ratio']
        return (dict(frames=len(x), repeated_fraction=ratio, drop_frozen=bool(drop), flags={k: runs(v) for k, v in masks.items()}, metrics=dict(blur_p01=float(np.quantile(x[:, 4], 0.01)), change_p99=float(np.quantile(x[:, 5], 0.99)))), hard, x[:, 5], long_repeat)

def cross_view_checks(changes, repeats, state_moving, policy=DEFAULT_POLICY):
    changes, repeats = (np.asarray(changes), np.asarray(repeats, bool))
    moving_view = np.any(changes > policy['visual_motion_mae'], axis=0)
    capture = np.any(repeats, axis=0) & moving_view & np.asarray(state_moving, bool)
    ambiguous = np.all(changes < policy['visual_static_mae'], axis=0) & state_moving
    return (capture, dict(frozen_view_during_motion=runs(capture), all_views_static_state_moves=runs(ambiguous)))
