"""Camera-only render projection sidecar; never steps or changes the scene."""
import json
import os
from pathlib import Path

FIELDS = ('cameraFocalLength', 'cameraAperture', 'cameraApertureOffset',
          'renderProductResolution', 'cameraModel', 'cameraViewTransform',
          'cameraProjection')

def _plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value

class CameraAudit:
    def __init__(self, env):
        self.root = None
        self.annotators = {}
        self.errors = {}
        bridge = os.environ.get('ROBODAWN_DESKTOP_BRIDGE')
        if not bridge or getattr(env, 'task_name', None) != 'pick_from_conveyor_by_image':
            return
        self.root = Path(bridge).parent / 'camera_projection_audit'
        try:
            self.root.mkdir(exist_ok=True)
            import omni.replicator.core as rep
            manager = env.camera_manager
            for index, name in enumerate(manager.camera_names[0]):
                if name not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                    continue
                try:
                    camera = manager.cameras[0][index]
                    render_path = camera.get_render_product_path()
                    if not render_path:
                        capture = env.capture_manager
                        if capture.num_envs != 1 or capture.camera_names[0][index] != name:
                            raise ValueError('tiled camera identity is not unambiguous')
                        render_path = capture.tiled_render_products[index].path
                    if not render_path:
                        raise ValueError('existing camera render product unavailable')
                    annotator = rep.AnnotatorRegistry.get_annotator('camera_params')
                    annotator.attach(render_path)
                    self.annotators[name] = annotator
                except Exception as exc:
                    self.errors[name] = str(exc)
        except Exception as exc:
            self.errors['initialization'] = str(exc)

    def capture(self, decision, native_step, calibration):
        if self.root is None:
            return
        payload = {'schema_version': 1, 'diagnostic_only': True,
                   'decision': decision, 'native_step': native_step,
                   'extra_physics_steps': 0, 'calibration_changed': False,
                   'recorded_calibration': calibration, 'cameras': {},
                   'errors': dict(self.errors)}
        for name, annotator in self.annotators.items():
            try:
                data = annotator.get_data()
                if not isinstance(data, dict):
                    raise ValueError('camera_params not ready or not a dictionary')
                payload['cameras'][name] = {key: _plain(data[key]) for key in FIELDS if key in data}
                if not payload['cameras'][name]:
                    payload['errors'][name] = 'camera_params empty; no extra render requested'
            except Exception as exc:
                payload['errors'][name] = str(exc)
        (self.root / ('decision_%04d.json' % decision)).write_text(
            json.dumps(_plain(payload), indent=2), encoding='utf-8')
        return payload

    def save_correction(self, calibration, provenance):
        (self.root / ('corrected_%04d.json' % provenance['decision'])).write_text(
            json.dumps({'calibration': calibration, 'provenance': provenance}, indent=2),
            encoding='utf-8')
