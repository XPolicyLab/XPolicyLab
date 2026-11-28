"""Dense 25 Hz observations, client-owned CD-LAM state, and MR0 rollout loops.

The shared constants/state helpers also serve model.py. This module deliberately
has no torch/model imports or initialization-time random sampling.
"""
from collections import OrderedDict
from uuid import uuid4
import cv2
import numpy as np
from XPolicyLab.utils.process_data import decode_image_bit
from numbers import Integral

SCHEMA = 'lam_mean1hz_stride4_v1'
FPS, STRIDE, HORIZON = 25, 4, 16
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
ALIASES = (('cam_head', 'head_camera', 'cam_high'),
           ('cam_left_wrist', 'left_camera'), ('cam_right_wrist', 'right_camera'))
FIRST_KEY = 'lam_episode_first'


def integer(value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'Expected integer >= {minimum}, got {value!r}')
    return int(value)


def frame_key(frame):
    return f'lam_source_{frame:08d}'


def initial_state():
    return dict(through_frame=-1, features=np.empty((0, 32), np.float32),
                partial_sum=np.zeros(32, np.float32), partial_count=0)


def validate_state(state):
    if not isinstance(state, dict):
        raise ValueError('Missing client-owned lam_state')
    through = integer(state.get('through_frame'), -1)
    features = np.asarray(state.get('features'), dtype=np.float32)
    partial = np.asarray(state.get('partial_sum'), dtype=np.float32)
    count = integer(state.get('partial_count'))
    seconds = max(through, 0) // FPS
    expected_count = max(0, through - max(STRIDE, seconds * FPS + 1) + 1)
    if features.shape != (seconds, 32) or partial.shape != (32,):
        raise ValueError('LAM token prefix or partial sum has invalid shape')
    if not np.isfinite(features).all() or not np.isfinite(partial).all():
        raise ValueError('LAM state must be finite')
    if count != expected_count or (count == 0 and np.any(partial != 0)):
        raise ValueError('LAM partial bucket differs from the processed frame')
    return dict(through_frame=through, features=features.copy(),
                partial_sum=partial.copy(), partial_count=count)


def source_indices(state, frame):
    frame = integer(frame)
    previous = state['through_frame']
    if (previous == -1 and frame != 0) or (previous >= 0 and frame != previous + HORIZON):
        raise ValueError('LAM decisions must start at frame 0 and advance by 16 steps')
    if frame == 0:
        return [0]
    first_end = max(STRIDE, previous + 1)
    return list(range(first_end - STRIDE, frame + 1))


def advance_state(state, frame, latents):
    """Append every newly completed pair, expose only complete-second means."""
    result = validate_state(state)
    source_indices(result, frame)
    ends = list(range(max(STRIDE, result['through_frame'] + 1), frame + 1))
    values = np.asarray(latents, np.float32).reshape(-1, 32)
    if values.shape != (len(ends), 32) or not np.isfinite(values).all():
        raise ValueError('Teacher output does not match every newly completed pair')
    completed = []
    for end, value in zip(ends, values):
        result['partial_sum'] += value
        result['partial_count'] += 1
        if end % FPS == 0:
            completed.append(result['partial_sum'] / result['partial_count'])
            result['partial_sum'] = np.zeros(32, np.float32)
            result['partial_count'] = 0
    if completed:
        result['features'] = np.concatenate((result['features'], np.asarray(completed, np.float32)))
    result['through_frame'] = frame
    return validate_state(result)


def model_memory(state):
    return dict(features=state['features'],
                end_times_sec=np.arange(1, len(state['features']) + 1, dtype=np.float32),
                current_time_sec=state['through_frame'] / FPS)


def camera(obs, aliases):
    for name in aliases:
        value = obs.get('vision', {}).get(name)
        if isinstance(value, dict):
            for key in ('color', 'rgb', 'colors', 'image'):
                if key in value:
                    return value[key]
        elif value is not None:
            return value
    raise ValueError(f'Missing camera {aliases}')


def lossless(image):
    # Historical first/teacher images were lossless PNG. Send arrays with
    # identical RGB values; never publish an unmarked encoded buffer.
    return np.ascontiguousarray(image).copy()


def jpeg(image):
    # Preserve the historical evaluation's exact JPEG-decoded current pixels.
    # This buffer is decoded locally and is NEVER stored or sent. All wire
    # images are RGB arrays (or incoming buffers from the benchmark).
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Could not reproduce current-camera JPEG pixels')
    return decode_image_bit(encoded).copy()


class LAMCache:
    def __init__(self):
        self.by_env = {}

    def reset_envs(self, indices):
        for index in indices:
            self.by_env.pop(integer(index), None)

    def capture(self, obs, frame):
        frame = integer(frame)
        index = integer(obs.get('env_idx', 0))
        bank = self.by_env.get(index)
        if bank is None:
            if frame != 0:
                raise ValueError('LAM episode must capture frame 0 first')
            bank = dict(episode_id=uuid4().hex, frames=OrderedDict(), first=None,
                        last=-1, state=initial_state(), pending=None)
        if frame == bank['last']:
            return
        if frame != bank['last'] + 1:
            raise ValueError('LAM needs every 25 Hz main-camera frame')
        if bank['pending'] is not None or frame > max(bank['state']['through_frame'], 0) + HORIZON:
            raise ValueError('Commit the previous LAM response before advancing the environment')
        rgb = camera(obs, ALIASES[0])
        if not (isinstance(rgb, np.ndarray) and rgb.ndim == 3):
            rgb = decode_image_bit(rgb)
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError('LAM capture requires RGB uint8 640x480 main camera')
        # At the required native size, this is exactly teacher center_crop_resize.
        bank['frames'][frame] = lossless(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_LINEAR))
        if frame == 0:
            bank['first'] = lossless(rgb)
        bank['last'] = frame
        self.by_env[index] = bank

    def request(self, obs, frame):
        frame = integer(frame)
        index = integer(obs.get('env_idx', 0))
        bank = self.by_env.get(index)
        if bank is None or bank['last'] != frame:
            raise ValueError('Capture through the current LAM decision frame before requesting')
        state = validate_state(bank['state'])
        selected = source_indices(state, frame)
        vision = {}
        for name, aliases in zip(CAMERAS, ALIASES):
            color = camera(obs, aliases)
            if isinstance(color, np.ndarray) and color.ndim == 3:
                color = jpeg(color)
            vision[name] = {'color': color}
        for n in selected:
            vision[frame_key(n)] = {'color': bank['frames'][n]}
        if frame:
            vision[FIRST_KEY] = {'color': bank['first']}
        result = dict(vision=vision, env_idx=index, state=obs['state'], instruction=obs['instruction'])
        result['lam_request'] = dict(schema=SCHEMA, episode_id=bank['episode_id'],
                                     frame_index=frame, timestamp_sec=frame/FPS,
                                     source_frame_indices=selected, state=state)
        bank['pending'] = frame
        return result

    def accept_batch(self, responses, indices):
        """Validate the whole batch before committing any environment state."""
        if not isinstance(responses, list) or len(responses) != len(indices) or len(set(indices)) != len(indices):
            raise ValueError('Invalid LAM response batch')
        pending = []
        for result, index in zip(responses, indices):
            bank = self.by_env[index]
            if (not isinstance(result, dict) or result.get('schema') != SCHEMA
                or result.get('episode_id') != bank['episode_id'] or result.get('env_idx') != index
                or bank['pending'] is None or result.get('frame_index') != bank['pending']):
                raise ValueError('LAM response belongs to another request, episode or environment')
            state = validate_state(result.get('state'))
            actions = result.get('actions')
            if state['through_frame'] != bank['pending'] or not isinstance(actions, list) or len(actions) != HORIZON:
                raise ValueError('Invalid LAM response cursor or action horizon')
            pending.append((bank, state, actions))
        for bank, state, _ in pending:
            bank['state'] = state
            bank['pending'] = None
            while bank['frames'] and next(iter(bank['frames'])) < state['through_frame'] - 3:
                bank['frames'].popitem(last=False)
        return [actions for _, _, actions in pending]

    def accept(self, result, index=0):
        return self.accept_batch([result], [index])[0]


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    cache = LAMCache()
    frame = 0
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        index = int(obs.get('env_idx', 0))
        cache.capture(obs, frame)
        packet = cache.request(obs, frame)
        response = model_client.call(func_name='predict_lam', obs=packet)
        actions = cache.accept(response, index)
        for offset, action in enumerate(actions):
            TASK_ENV.take_action(action)
            frame += 1
            if TASK_ENV.is_episode_end() or offset == 15:
                break
            cache.capture(TASK_ENV.get_obs(), frame)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    cache, frames = LAMCache(), {}

    def observations(indices):
        obs_list = TASK_ENV.get_obs_batch(indices)
        if len(obs_list) != len(indices):
            raise ValueError('LAM observation batch size differs from active environments')
        result = []
        for index, obs in zip(indices, obs_list):
            if int(obs.get('env_idx', index)) != index:
                raise ValueError('LAM observation env_idx order differs from requested batch')
            result.append(dict(obs, env_idx=index))
        return result

    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        payloads = []
        for index, obs in zip(indices, observations(indices)):
            frame = frames.setdefault(index, 0)
            cache.capture(obs, frame)
            payloads.append(cache.request(obs, frame))
        responses = model_client.call(func_name='predict_lam', obs=payloads)
        chunks = cache.accept_batch(responses, indices)
        for index in indices:
            state = cache.by_env[index]['state']
            if len(state['features']) != frames[index] // 25:
                raise ValueError('LAM complete-second history length differs from frame_index // 25')
        for offset in range(16):
            TASK_ENV.take_action_batch([chunk[offset] for chunk in chunks], indices)
            for index in indices:
                frames[index] += 1
            # RoboDojo refreshes per-env end flags in is_episode_end().
            episode_end = TASK_ENV.is_episode_end()
            running = set(TASK_ENV.get_running_env_idx_list())
            finished = [index for index in indices if index not in running]
            cache.reset_envs(finished)
            for index in finished:
                frames.pop(index, None)
            keep = [i for i, index in enumerate(indices) if index in running]
            chunks = [chunks[i] for i in keep]
            indices = [indices[i] for i in keep]
            if episode_end or offset == 15:
                break
            for index, obs in zip(indices, observations(indices)):
                cache.capture(obs, frames[index])
