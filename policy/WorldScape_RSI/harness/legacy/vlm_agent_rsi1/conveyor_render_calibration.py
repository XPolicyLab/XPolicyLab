"""Validated per-observation rendering calibration, without changing cameras."""
import copy
import numpy as np

VERSION = 'conveyor-render-projection-v1'

def derive(payload):
    original = payload.get('recorded_calibration', {})
    corrected = {}
    provenance = {'version': VERSION, 'decision': payload['decision'],
                  'native_step': payload['native_step'],
                  'source': 'existing tiled render product camera_params',
                  'original_preserved_in': 'camera_projection_audit/decision_%04d.json' % payload['decision'],
                  'camera_settings_changed': False, 'extra_physics_steps': 0,
                  'cameras': {}}
    for name, raw in original.items():
        try:
            c = payload['cameras'][name]
            w, h = c['renderProductResolution']
            p = np.asarray(c['cameraProjection'], dtype=float).reshape(4, 4)
            v = np.asarray(c['cameraViewTransform'], dtype=float).reshape(4, 4)
            e = np.asarray(raw['extrinsic_matrix'], dtype=float)
            k = np.asarray(raw['intrinsic_matrix'], dtype=float)
            assert c['cameraModel'] == 'pinhole' and w == 640 and h == 480
            assert np.isfinite(p).all() and np.isfinite(v).all()
            assert p[0, 0] > 0 and p[1, 1] > 0
            assert np.allclose(c['cameraApertureOffset'], [0, 0], atol=1e-8)
            assert np.allclose(k[:2, 2], [w/2, h/2], atol=1e-5)
            error = float(np.max(np.abs(np.linalg.inv(v).T - e)))
            assert error < 1e-4, 'render pose does not match observation'
            new = copy.deepcopy(raw)
            new['intrinsic_matrix'][0][0] = float(p[0, 0] * w / 2)
            new['intrinsic_matrix'][1][1] = float(p[1, 1] * h / 2)
            corrected[name] = new
            provenance['cameras'][name] = {'status': 'measured_render_projection',
                'resolution': [w, h], 'extrinsic_max_error': error,
                'original_intrinsic_matrix': raw['intrinsic_matrix'],
                'corrected_intrinsic_matrix': new['intrinsic_matrix']}
        except (KeyError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
            # First attach may produce all zeros until the next normal render.
            # Omit unavailable geometry rather than expose a known inaccurate K.
            provenance['cameras'][name] = {'status': 'unavailable_omitted',
                'reason': str(exc) or 'unready or unsupported projection'}
    return corrected, provenance
