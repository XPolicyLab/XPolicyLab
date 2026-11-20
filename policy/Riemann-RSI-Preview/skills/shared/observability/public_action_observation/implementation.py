"""Current RGB, own arm/EE pose and native singleton normalized feedback; never real finger width."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
import numpy as np
from skills.shared.geometry.x5_public_frame_model.implementation import ARMS, CAMERAS

STATE_KEYS = tuple(arm + suffix for arm in ARMS for suffix in (
    '_arm_joint_state', '_ee_pose', '_ee_joint_state'))
MAX_OBSERVATION_BYTES = 4_000_000


def plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False))


@dataclass(frozen=True)
class Observation:
    observation_id: int
    physical_step: int
    state: dict
    vision: dict
    evidence: dict
    previous_summary: str = ''

    @classmethod
    def create(cls, state, vision, step, previous_summary=''):
        if type(step) is not int or step < 0 or step > 1050:
            raise ValueError('Invalid native control step')
        if set(state) != set(STATE_KEYS):
            raise ValueError('Policy state allowlist mismatch')
        if set(vision) != set(CAMERAS):
            raise ValueError('Exactly current head, left wrist and right wrist RGB views are mandatory')
        if not isinstance(previous_summary, str) or len(previous_summary) > 5000:
            raise ValueError('Input text memory exceeds 5000 characters')
        copied_state = {}
        for key in STATE_KEYS:
            expected = 6 if key.endswith('_arm_joint_state') else 7 if key.endswith('_ee_pose') else 1
            value = np.asarray(state[key], float)
            if value.shape != (expected,) or not np.isfinite(value).all():
                raise ValueError('Invalid current measured robot state: ' + key)
            if key.endswith('_ee_joint_state') and not 0 <= value[0] <= 1:
                raise ValueError('Normalized public gripper feedback must be in [0,1]')
            copied_state[key] = value.copy()
        copied_images = {}; evidence = {}; size = len(json.dumps(plain(copied_state)).encode()) + len(previous_summary.encode())
        for cam in CAMERAS:
            image = np.asarray(vision[cam])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f'{cam} must be unmodified 640x480 RGB uint8')
            image = np.ascontiguousarray(image).copy()
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            copied_images[cam] = image
            evidence[cam] = dict(observation_id=step, physical_step=step, camera=cam,
                                 sha256=digest, width=640, height=480, source='native_RGB')
            size += image.nbytes
        if size > MAX_OBSERVATION_BYTES:
            raise ValueError('Current observation request exceeds byte budget; do not drop a camera')
        evidence['preflight'] = dict(image_count=3, historical_images=0, image_bytes=size,
                                     memory_characters=len(previous_summary), API_context_tokens=None,
                                     controller='deterministic; no model request or conversation history')
        return cls(step, step, copied_state, copied_images, evidence, previous_summary)

    def bind(self, action):
        return dict(observation_id=self.observation_id, physical_step=self.physical_step,
                    action=plain(action))


def validate_bound_action(bound, latest):
    if (bound['observation_id'], bound['physical_step']) != (latest.observation_id, latest.physical_step):
        raise ValueError('Refusing stale action')
    expected = {arm + suffix for arm in ARMS for suffix in ('_arm_joint_state', '_ee_joint_state')}
    if set(bound['action']) != expected:
        raise ValueError('Joint action allowlist mismatch')
    for key, value in bound['action'].items():
        value = np.asarray(value, float)
        if value.shape != ((6,) if key.endswith('_arm_joint_state') else (1,)) or not np.isfinite(value).all():
            raise ValueError('Invalid joint action')
        if key.endswith('_ee_joint_state') and not 0 <= value[0] <= 1:
            raise ValueError('Invalid commanded gripper opening')
    return bound['action']


def normalized_feedback(state, arm):
    feedback = np.asarray(state[arm + '_ee_joint_state'], float)
    if feedback.shape != (1,) or not np.isfinite(feedback).all() or not 0 <= feedback[0] <= 1:
        raise ValueError('Expected the native singleton normalized gripper observation')
    return float(feedback[0])


def public_finger_openings(state, arm, rig):
    normalized_feedback(state, arm)
    return np.asarray(rig.maximum_openings, float).copy()


def opening_hypotheses(state, arm, rig):
    feedback = normalized_feedback(state, arm)
    low, high = rig.primary_limits
    nominal = abs(low + feedback * (high - low))
    return (np.zeros(2), np.full(2, nominal), np.asarray(rig.maximum_openings).copy())


def payload_evidence(context, arm, phase):
    feedback = normalized_feedback(context.obs.state, arm)
    evidence = dict(state=None, phase=phase, observation_id=context.obs.observation_id,
                    normalized_feedback=feedback, commanded_opening=context.grips[arm],
                    measured_finger_pair=False, measured_width_m=None,
                    scalar_is_prev_control_not_contact=True, reason='private width predicate is unknown')
    if context.target is None or context.grips[arm] != 0.:
        return evidence
    tools = getattr(getattr(context, 'reader', None), 'tools', None)
    if tools is None or not hasattr(tools, 'verify_digit_identity'):
        return evidence
    from skills.shared.geometry.x5_public_frame_model.implementation import rotation
    wrist = context.obs.state[arm + '_ee_pose']
    expected = np.asarray(wrist[:3]) + rotation(wrist[3:])[:, 0] * .155
    from skills.shared.perception.material_glyph_outline.implementation import RecognitionNeedsObservation
    try:
        _, report = tools.verify_digit_identity(context.obs, context.frame, context.target, expected[:2], placed=False)
    except RecognitionNeedsObservation as error:
        report = error.report
    except (ValueError, RuntimeError):
        return evidence
    if report.get('observation_id') != context.obs.observation_id:
        return evidence
    confirmed = [item['detection'] for item in report.get('objects', [])
                 if item.get('status') == 'confirmed' and item.get('detection', {}).get('digit') == context.target]
    if len(confirmed) == 1:
        distance = float(np.linalg.norm(np.asarray(confirmed[0]['xy']) - expected[:2]))
        evidence.update(current_rgb_digit_xy=np.asarray(confirmed[0]['xy']).tolist(), tool_distance_m=distance)
        if distance > .055:
            evidence.update(state=False, reason='independent current RGB digit is beyond original release-verification displacement tolerance from the public tool')
    return evidence
