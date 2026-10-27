import os
from typing import Any
from go3_nano_preview.sft.data.unified_dataset.base_dataset import EpisodeSampleSelection, MixBase
from go3_nano_preview.sft.data.unified_dataset.data_utils import get_video_info
import numpy as np
import random
import torch
import gc
import h5py
from func_timeout import func_set_timeout, FunctionTimedOut
import pyarrow.parquet as pq
import pandas as pd
import time
from go3_nano_preview.sft.data.unified_dataset.extract_step_caption_union_interval import compute_step_caption_union_ranges

def _step_speed_metric(entry):
    eef = entry.get('eef_speed') if isinstance(entry, dict) else None
    if not isinstance(eef, dict) or eef.get('status') != 'ok':
        return None
    moving_mean = eef.get('moving_mean')
    return float(moving_mean) if moving_mean is not None else 0.0

def _join_step_captions(captions):
    if len(captions) == 1:
        return captions[0]
    parts = []
    for (i, c) in enumerate(captions):
        c = c.rstrip()
        if i > 0:
            if c and c[0].isupper():
                c = c[0].lower() + c[1:]
        if i < len(captions) - 1:
            if not c.endswith('.'):
                c += '.'
        parts.append(c)
    return ' And then, '.join(parts)

def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for (s, e) in intervals[1:]:
        (ps, pe) = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _allowed_segments(video_lo, video_hi, forbidden):
    if video_hi < video_lo:
        return []
    clean = []
    for (s, e) in forbidden:
        if e < video_lo or s > video_hi:
            continue
        s = max(s, video_lo)
        e = min(e, video_hi)
        if s <= e:
            clean.append((s, e))
    merged = _merge_intervals(clean)
    allowed = []
    cur = video_lo
    for (s, e) in merged:
        if cur <= s - 1:
            allowed.append((cur, s - 1))
        cur = e + 1
    if cur <= video_hi:
        allowed.append((cur, video_hi))
    return allowed

def _pick_t0_weighted(py_rng, allowed):
    total = sum((b - a + 1 for (a, b) in allowed))
    pick = py_rng.randrange(total)
    acc = 0
    for (a, b) in allowed:
        w = b - a + 1
        if acc + w > pick:
            t0 = py_rng.randint(a, b)
            return (t0, (a, b))
        acc += w
    return (allowed[0][0], allowed[0])

def get_variable_stride_indices_general(L: int, device=None):
    idx_list = []
    end0 = min(50, L)
    if end0 > 0:
        idx_list.append(torch.arange(0, end0))
    start = 50
    stride = 2
    while start < L:
        end = min(start + 50 * stride, L)
        first = start + stride - 1
        if first < end:
            seg_idx = torch.arange(first, end, step=stride)
            idx_list.append(seg_idx)
        start = end
        stride *= 2
    idx = torch.cat(idx_list) if idx_list else torch.empty(0, dtype=torch.long)
    return idx.to(device)

class MixMultiFrame(MixBase):

    def __init__(self, jsonl_path_list, video_folder_list, dataset_source_list, dataset_name_list, action_seq_length=2048, continious_vid_length=9, continious_action_length=54, sparse_length=9, min_gap=60, choose_start_end=False, **kwargs):
        super().__init__(jsonl_path_list, video_folder_list, dataset_source_list, dataset_name_list, **kwargs)
        self.strip_goal_frame = kwargs.get('strip_goal_frame', False)
        self.down_ratio = continious_action_length // continious_vid_length
        assert self.down_ratio * continious_vid_length == continious_action_length
        self.continious_vid_length = continious_vid_length
        self.continious_action_length = continious_action_length
        self.sparse_length = sparse_length
        self.action_seq_length = action_seq_length
        self.min_gap = min_gap
        self.choose_start_end = choose_start_end
        self.next_step_prob = kwargs.get('next_step_prob', 0.0)
        self.single_step_sampling = bool(kwargs.get('single_step_sampling', False))
        self.goal_end_jitter = int(kwargs.get('goal_end_jitter', 0))
        if self.goal_end_jitter > 0:
            assert self.min_gap > self.goal_end_jitter, f'goal_end_jitter={self.goal_end_jitter} requires min_gap > goal_end_jitter (got min_gap={self.min_gap})'
        self.force_goal_on_empty_caption = kwargs.get('force_goal_on_empty_caption', False)
        print('Dataset __init__ decode_type:', self.decode_type)
        self.sampling_statagy = kwargs.get('sampling_statagy', 'msth')
        self.video_length = kwargs.get('video_length', 6)

    def get_total_timesteps(self, video_dict):
        action_length = video_dict['action_frames']
        t = video_dict['video_frames']
        if action_length != t:
            if abs(action_length - t) <= 10:
                common_len = min(action_length, t)
                action_length = common_len
                t = common_len
            assert action_length == t
        return t

    def get_step_start(self, video_slices, action_slices):
        return action_slices[self.n_previous_action]

    def get_batch_new(self, idx, **kwargs):
        results = super().get_batch_new(idx, **kwargs)
        if getattr(self, 'strip_goal_frame', False) and 'video' in results and (results['video'].shape[2] >= 3):
            _keep = [0] + list(range(2, results['video'].shape[2]))
            results['video'] = results['video'][:, :, _keep]
            if 'video_slices' in results and torch.is_tensor(results['video_slices']):
                results['video_slices'] = results['video_slices'][_keep]
        if getattr(self, 'mix_captions', False):
            goal_mode = getattr(self, '_mix_caption_goal_mode', 'task')
            step_captions = getattr(self, '_step_goal_captions', [])
            if goal_mode == 'step':
                results['goal_mode'] = 'next-step' if len(step_captions) > 1 else 'step'
            else:
                results['goal_mode'] = 'task'
            results['raw_caption'] = _join_step_captions(step_captions) if goal_mode == 'step' else results.get('raw_task_caption', results.get('task_caption', ''))
            results['final_caption'] = results.get('caption', '')
            results['goal_end_frame'] = results['video_slices'][1]
        if kwargs.get('goal_mode'):
            results['goal_mode'] = 'goal-frame'
        if self.force_goal_on_empty_caption and 'goal_cond' in results:
            caption = results.get('caption', '')
            if isinstance(caption, str) and caption.strip().lower() in self._empty_caption_values:
                results['goal_cond'] = torch.tensor(True)
                results['goal_mode'] = 'goal-frame-empty-caption'
        return results

    def get_true_start_end(self, video_dict, end):
        if not video_dict['step_caption']:
            return (0, end)
        low = 9999
        hi = 0
        for l in video_dict['step_caption']:
            low = min(low, l['start_frame'])
            hi = max(hi, l['end_frame'])
        return (low, hi)

    def compute_episode_sampling_weight(self, episode_index: int, episode: dict[str, Any]) -> float:
        if self.mix_captions:
            ranges = compute_step_caption_union_ranges(episode, self.dataset_name_list[episode_index], self.min_gap)
            return sum((length for (_, _, length) in ranges))
        return self._cal_sample_index_allowed_weight(episode)

    def _cal_sample_index_allowed_weight(self, video_dict):
        total_frames = self._sampling_total_frames(video_dict)
        if total_frames <= 0:
            return 0.0
        end = total_frames - 1
        (start, end) = self._sampling_start_end(video_dict, end)
        forbidden = self._sampling_forbidden_intervals(video_dict)
        min_gap = int(getattr(self, 'min_gap', 0))
        allowed = self._sampling_allowed_segments(start, end, forbidden)
        allowed = [(a, b) for (a, b) in allowed if b - a >= min_gap]
        return sum((b - a + 1 for (a, b) in allowed))

    def _sampling_total_frames(self, video_dict):
        video_frames = int(video_dict.get('video_frames', 0) or 0)
        action_frames = int(video_dict.get('action_frames', video_frames) or video_frames)
        if video_frames <= 0:
            return max(action_frames, 0)
        if action_frames <= 0:
            return video_frames
        if action_frames != video_frames and abs(action_frames - video_frames) <= 10:
            return min(action_frames, video_frames)
        return min(action_frames, video_frames)

    def _sampling_start_end(self, video_dict, end):
        steps = video_dict.get('step_caption')
        if not steps:
            return (0, end)
        starts = []
        ends = []
        for step in steps:
            try:
                starts.append(int(step['start_frame']))
                ends.append(int(step['end_frame']))
            except (KeyError, TypeError, ValueError):
                continue
        if not starts or not ends:
            return (0, end)
        return (max(0, min(starts)), min(end, max(ends)))

    def _sampling_forbidden_intervals(self, video_dict):
        if not self.use_adc:
            return []
        forbidden = []
        for item in video_dict.get('key_frame', {}).get('dual', []):
            use_item = False
            if 'comment' in item and item['comment'] == 'Error Frame':
                use_item = True
            elif 'frame_type_name' in item and item['frame_type_name'] == 'Error Frame':
                use_item = True
            elif 'comment' not in item and 'frame_type_name' not in item:
                use_item = True
            if not use_item:
                continue
            try:
                start = int(item.get('start', 0))
                end = int(item.get('end', -1))
            except (TypeError, ValueError):
                continue
            if start <= end:
                forbidden.append((start, end))
        return forbidden

    def _sampling_allowed_segments(self, video_lo, video_hi, forbidden):
        if video_hi < video_lo:
            return []
        clipped = []
        for (start, end) in forbidden:
            if end < video_lo or start > video_hi:
                continue
            start = max(start, video_lo)
            end = min(end, video_hi)
            if start <= end:
                clipped.append((start, end))
        if not clipped:
            return [(video_lo, video_hi)]
        clipped.sort(key=lambda x: x[0])
        merged = [list(clipped[0])]
        for (start, end) in clipped[1:]:
            prev = merged[-1]
            if start <= prev[1] + 1:
                prev[1] = max(prev[1], end)
            else:
                merged.append([start, end])
        allowed = []
        cur = video_lo
        for (start, end) in merged:
            if cur <= start - 1:
                allowed.append((cur, start - 1))
            cur = end + 1
        if cur <= video_hi:
            allowed.append((cur, video_hi))
        return allowed

    def sample_episode_frames(self, end: int, episode_index: int, episode: dict[str, Any], *, goal_mode: bool, datasource: str | None=None) -> EpisodeSampleSelection:
        if self.mix_captions and (not goal_mode):
            selection = self.cal_sample_index_mix_captions(end, episode_index, episode, datasource=datasource)
        else:
            selection = self.cal_sample_index(end, episode_index, episode, goal_mode=goal_mode, datasource=datasource)
        (end_frames, end_frames_action, video_slices, action_slices) = selection
        return EpisodeSampleSelection(end_frames=int(end_frames), end_frames_action=int(end_frames_action), video_slices=tuple((int(index) for index in video_slices)), action_slices=tuple((int(index) for index in action_slices)), sample_metadata={})

    def cal_sample_index(self, end, idx, video_dict, *, goal_mode=False, datasource=None):
        py_rng = random.Random(idx) if self.fix_timestep else random
        if 'step_caption' in video_dict:
            (start, end) = self.get_true_start_end(video_dict, end)
        else:
            (start, end) = (0, end)
        if self.use_adc:
            dual_fake = video_dict.get('key_frame', {}).get('dual', [])
            dual = []
            for item in dual_fake:
                if 'comment' in item and item['comment'] == 'Error Frame':
                    dual.append(item)
                elif 'frame_type_name' in item and item['frame_type_name'] == 'Error Frame':
                    dual.append(item)
                elif 'comment' not in item and 'frame_type_name' not in item:
                    dual.append(item)
        else:
            dual = []
        forbidden = []
        for d in dual:
            s = int(d.get('start', 0))
            e = int(d.get('end', -1))
            if s <= e:
                forbidden.append((s, e))
        error_frame = []
        error_items = video_dict.get('key_frame', {}).get('error', [])
        for item in error_items:
            s = int(item.get('start', 0))
            e = int(item.get('end', -1))
            if s <= e:
                error_frame.append((s, e))
        error_frame.sort(key=lambda x: (x[0], x[1]))
        allowed = _allowed_segments(start, end, forbidden)
        allowed = [(a, b) for (a, b) in allowed if b - a >= self.min_gap]
        if not allowed:
            raise ValueError('无可用允许区间，无法采样。')
        desired = self.action_seq_length
        use_step_goal = False
        tries = 32
        for _ in range(tries):
            (t0, (a, b)) = _pick_t0_weighted(py_rng, allowed)
            t0_in_error = False
            for (s, e) in error_frame:
                if s <= t0 and t0 <= e:
                    b = min(b, e)
                    t0_in_error = True
                    break
            t1_min = t0 + self.min_gap
            t1_max = min(t0 + desired, b)
            if not t0_in_error:
                for (s, e) in error_frame:
                    if s >= t0 and s <= b:
                        t1_max = min(t1_max, e)
                        break
                    if s > b:
                        break
            if self.mix_captions and (not goal_mode):
                if self.step_goal_prob > 0 and 'step_caption' in video_dict and video_dict['step_caption'] and (py_rng.random() < self.step_goal_prob) or ('task_caption' in video_dict and video_dict['task_caption'] in ['None', '']):
                    steps = video_dict['step_caption']
                    for (i, entry) in enumerate(steps):
                        if entry['start_frame'] <= t0 < entry['end_frame'] and t1_max >= entry['end_frame']:
                            use_next = self.next_step_prob > 0 and i + 1 < len(steps) and (py_rng.random() < self.next_step_prob)
                            if use_next and steps[i + 1]['end_frame'] < t1_max:
                                next_entry = steps[i + 1]
                                if next_entry['end_frame'] - t0 >= self.min_gap:
                                    t1 = next_entry['end_frame']
                                    use_step_goal = True
                                    self._step_goal_captions = [entry['action_text'], next_entry['action_text']]
                                    self._step_goal_speed_value = None
                            if not use_step_goal:
                                if entry['end_frame'] - t0 >= self.min_gap:
                                    t1 = entry['end_frame']
                                    use_step_goal = True
                                    self._step_goal_captions = [entry['action_text']]
                                    self._step_goal_speed_value = _step_speed_metric(entry)
                            break
                    if not use_step_goal:
                        continue
                if not use_step_goal:
                    t1 = t1_max
                break
            elif t1_min <= t1_max:
                t1 = py_rng.randint(t1_min, t1_max)
                break
        else:
            raise ValueError
            (a, b) = max(allowed, key=lambda x: x[1] - x[0])
            t0 = t1 = a
        if not use_step_goal:
            self._step_goal_speed_value = None
        self._use_step_goal = use_step_goal
        self._mix_caption_goal_mode = 'step' if use_step_goal else 'task'
        if self.choose_start_end:
            (t0, t1) = (min(start + 120, end - 31), end - 1)
        (video_slices, action_slices) = self.get_slices_from_start_end(t0, t1, end, self.continious_action_length, self.continious_vid_length, self.sparse_length, self.down_ratio, self.video_length, datasource)
        end_frames = 0
        end_frames_action = 0
        if self.ignore_seek:
            video_slices = [t0, t1, t1]
            end_frames = 1
        return (end_frames, end_frames_action, video_slices, action_slices)

    def cal_sample_index_mix_captions(self, end, idx, video_dict, datasource=None):
        py_rng = random.Random(idx) if self.fix_timestep else random
        self._step_goal_captions = []
        self._step_goal_speed_value = None
        computed_ranges = compute_step_caption_union_ranges(video_dict, self.dataset_name_list[idx], self.min_gap)
        steps = video_dict['step_caption']
        single_step_mode = self.single_step_sampling and bool(steps)
        dense_step_mode = self.use_dense_prompt and bool(steps)
        if single_step_mode:
            sampling_ranges = []
            for (step, (range_start, range_end, _)) in zip(steps, computed_ranges):
                step_start = max(int(step['start_frame']), range_start)
                step_end = int(step['end_frame'])
                valid_end = min(range_end, step_end - 1 - self.min_gap)
                length = max(0, valid_end - step_start + 1)
                sampling_ranges.append((step_start, valid_end, length))
        else:
            sampling_ranges = computed_ranges
        row_lengths = [length for (_, _, length) in sampling_ranges]
        local_step_idx = py_rng.choices(range(len(sampling_ranges)), weights=row_lengths, k=1)[0]
        self._sampled_step_index = int(local_step_idx)
        (interval_start, interval_end, _) = sampling_ranges[local_step_idx]
        if single_step_mode or len(steps) == 0:
            t0 = py_rng.randint(interval_start, interval_end)
        else:
            current_step_start = interval_start
            if steps:
                current_step_start = int(steps[local_step_idx]['start_frame'])
            if interval_start < current_step_start <= interval_end:
                left_segment = (interval_start, current_step_start - 1)
                right_segment = (current_step_start, interval_end)
                segment = py_rng.choices([left_segment, right_segment], weights=[self.next_step_prob, 1 - self.next_step_prob], k=1)[0]
                t0 = py_rng.randint(segment[0], segment[1])
            else:
                t0 = py_rng.randint(interval_start, interval_end)
        if single_step_mode:
            t1 = int(steps[local_step_idx]['end_frame']) - 1
        else:
            t1 = int(steps[local_step_idx]['end_frame']) if len(steps) > 0 else video_dict['video_frames'] - 1
        if single_step_mode:
            self._step_goal_captions = [self._select_step_prompt(steps[local_step_idx], py_rng, frame_idx=t0)]
        else:
            for step_idx in range(local_step_idx, -1, -1):
                (prev_start, prev_end, _) = computed_ranges[step_idx]
                if not prev_start <= t0 <= prev_end:
                    break
                if len(steps) == 0:
                    step_caption = video_dict['task_caption']
                    if len(self._step_goal_captions) == 0:
                        self._step_goal_captions.insert(0, step_caption)
                else:
                    step_caption = str(steps[step_idx]['action_text'])
                    self._step_goal_captions.insert(0, step_caption)
        if len(steps) > 0 and len(self._step_goal_captions) == 1:
            self._step_goal_speed_value = _step_speed_metric(steps[local_step_idx])
        task_caption = video_dict['task_caption']
        if single_step_mode or task_caption in ['', 'None']:
            use_step_goal = True
        elif py_rng.random() < self.step_goal_prob:
            use_step_goal = True
        else:
            use_step_goal = False
            self._step_goal_captions = [task_caption]
            self._step_goal_speed_value = None
        self._use_step_goal = use_step_goal
        self._mix_caption_goal_mode = 'step' if use_step_goal else 'task'
        if self.goal_end_jitter > 0:
            t1 -= py_rng.randint(0, self.goal_end_jitter)
        (video_slices, action_slices) = self.get_slices_from_start_end(t0, t1, end, self.continious_action_length, self.continious_vid_length, self.sparse_length, self.down_ratio, self.video_length, datasource)
        end_frames = 0
        end_frames_action = 0
        if self.ignore_seek:
            video_slices = [t0, t1, t1]
            end_frames = 1
        return (end_frames, end_frames_action, video_slices, action_slices)

    def get_slices_from_start_end(self, t0, t1, end, continious_action_length, continious_vid_length, sparse_length, down_ratio, video_length=None, datasource=None):
        valid_length = t1 - t0
        if getattr(self, 'sampling_statagy', 'msth') == 'msth':
            vid_continious_action_length = continious_action_length
            vid_down_ratio = down_ratio
            if valid_length > vid_continious_action_length:
                video_slices = [t0, t1] + list(range(t0 + 1, t0 + vid_continious_action_length + 1))[vid_down_ratio - 1::vid_down_ratio] + torch.linspace(t0 + vid_continious_action_length, t1, steps=sparse_length + 1)[1:].round().int().tolist()
                action_slices = [t0] + list(range(t0 + 1, t0 + continious_action_length + 1)) + torch.linspace(t0 + continious_action_length, t1, steps=sparse_length + 1)[1:].round().int().tolist() + [t1]
            else:
                continious_idx = list(range(t0 + 1, t1))[vid_down_ratio - 1::vid_down_ratio]
                video_slices = [t0, t1] + continious_idx + [t1] * (continious_vid_length + sparse_length - len(continious_idx))
                action_slices = [t0] + list(range(t0 + 1, t1 + 1)) + [t1] * (continious_action_length + sparse_length - valid_length) + [t1]
        elif getattr(self, 'sampling_statagy', 'msth') == 'uniform':
            assert video_length is not None
            assert not self.return_action
            video_slices = [t0, t1] + torch.linspace(t0, t1, steps=video_length + 1)[1:].round().int().tolist()
            action_slices = [t0] + torch.linspace(t0, t1, steps=video_length + 1)[1:].round().int().tolist()
        else:
            raise NotImplementedError
        return (video_slices, action_slices)

    def cal_continuous_index(self, cur_idx, action_chunk, end, *, step_caption):
        t0 = min(cur_idx, max(0, end - action_chunk))
        for step in step_caption:
            if step['start_frame'] <= t0 < step['end_frame']:
                t1 = min(int(step['end_frame']), end)
                break
        else:
            if not step_caption:
                t1 = t0 + 100
            else:
                raise ValueError(f'No step_caption interval contains validation start frame {t0}')
        (video_slices, action_slices) = self.get_slices_from_start_end(t0, t1, end, self.continious_action_length, self.continious_vid_length, self.sparse_length, self.down_ratio, self.video_length)
        return (0, 0, video_slices, action_slices)
