import json
import os
import numpy as np
from tqdm import tqdm

def _get_task_caption(record, dataset_name):
    clip = record['clip']
    task_caption = record['task_caption'].strip()
    if task_caption == '':
        raise ValueError(f'task_caption is empty or whitespace-only for clip: {clip}, dataset: {dataset_name}')
    return task_caption

def _get_forbidden_intervals(record, frame_count):
    dual_fake = record.get('key_frame', {}).get('dual', [])
    forbidden = []
    for item in dual_fake:
        if item.get('frame_type_name') != 'Error Frame':
            continue
        start = max(int(item['start']), 0)
        end = min(int(item['end']), frame_count - 1)
        if start <= end:
            forbidden.append((start, end))
    return forbidden

def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for (start, end) in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for (start, end) in merged]

def compute_step_caption_union_ranges(record, dataset_name, min_gap):
    frame_count = int(record['video_frames'])
    steps = _normalize_steps(record, dataset_name, frame_count)
    forbidden = _merge_intervals(_get_forbidden_intervals(record, frame_count))
    ranges = []
    for (start, end) in steps:
        if any((forbidden_start <= end <= forbidden_end for (forbidden_start, forbidden_end) in forbidden)):
            ranges.append((-1, -1, 0))
            continue
        previous_forbidden_end = max((forbidden_end for (_, forbidden_end) in forbidden if forbidden_end < end), default=-1)
        valid_start = max(start, previous_forbidden_end + 1)
        valid_end = end - int(min_gap)
        ranges.append((-1, -1, 0) if valid_start > valid_end else (valid_start, valid_end, valid_end - valid_start + 1))
    return ranges

def _normalize_steps(record, dataset_name, frame_count):
    steps = record['step_caption']
    bounds = []
    if len(steps) == 0:
        _get_task_caption(record, dataset_name)
        bounds.append((0, frame_count - 1))
        return bounds
    prev_end = None
    current_block_start = None
    for step in steps:
        raw_start = max(int(step['start_frame']), 0)
        end = min(int(step['end_frame']), frame_count - 1)
        if raw_start > end:
            end = raw_start
        if prev_end is None:
            current_block_start = raw_start
        elif raw_start > prev_end + 1:
            current_block_start = raw_start
        start = current_block_start
        bounds.append((start, end))
        prev_end = end
    return bounds

def _count_step_bounds(record):
    steps = record['step_caption']
    return 1 if len(steps) == 0 else len(steps)

def _count_forbidden_intervals(record):
    frame_count = int(record['video_frames'])
    return len(_merge_intervals(_get_forbidden_intervals(record, frame_count)))

def _fill_row_arrays(record, dataset_name, row_idx, write_idx, forbidden_starts_by_row, forbidden_ends_by_row, step_bounds, step_rows):
    frame_count = int(record['video_frames'])
    step_bounds_list = _normalize_steps(record, dataset_name, frame_count)
    forbidden = _merge_intervals(_get_forbidden_intervals(record, frame_count))
    forbidden_capacity = forbidden_starts_by_row.shape[1]
    if len(forbidden) > forbidden_capacity:
        raise ValueError(f"forbidden count {len(forbidden)} exceeds allocated capacity {forbidden_capacity} for clip: {record['clip']}")
    for (forbidden_idx, (start, end)) in enumerate(forbidden):
        forbidden_starts_by_row[row_idx, forbidden_idx] = start
        forbidden_ends_by_row[row_idx, forbidden_idx] = end
    next_write_idx = write_idx + len(step_bounds_list)
    for (local_step_idx, (start, end)) in enumerate(step_bounds_list):
        flat_step_idx = write_idx + local_step_idx
        step_bounds[flat_step_idx, 0, 0] = start
        step_bounds[flat_step_idx, 1, 0] = end
        step_rows[flat_step_idx, 0] = row_idx
    return next_write_idx

def _find_valid_ranges_for_all_steps(step_bounds, step_rows, forbidden_starts_by_row, forbidden_ends_by_row, min_gap):
    if step_bounds.shape[0] == 0:
        return np.empty((0, 2, 1), dtype=np.int32)
    rows = step_rows[:, 0]
    step_starts = step_bounds[:, 0, 0]
    step_ends = step_bounds[:, 1, 0]
    valid_ends = step_ends - int(min_gap)
    row_forbidden_starts = forbidden_starts_by_row[rows]
    row_forbidden_ends = forbidden_ends_by_row[rows]
    inside_candidates = (row_forbidden_starts <= step_ends[:, None]) & (step_ends[:, None] <= row_forbidden_ends)
    inside_mask = np.any(inside_candidates, axis=1)
    valid_ranges = np.full_like(step_bounds, np.int32(-1))
    keep_mask = ~inside_mask
    if not np.any(keep_mask):
        return valid_ranges
    kept_step_starts = step_starts[keep_mask]
    kept_step_ends = step_ends[keep_mask]
    kept_valid_ends = valid_ends[keep_mask]
    kept_row_forbidden_ends = row_forbidden_ends[keep_mask]
    previous_end_candidates = np.where(kept_row_forbidden_ends < kept_step_ends[:, None], kept_row_forbidden_ends, np.int32(-1))
    prev_forbidden_end_plus_one = previous_end_candidates.max(axis=1) + 1
    kept_valid_starts = np.maximum(kept_step_starts, prev_forbidden_end_plus_one)
    non_empty_mask = kept_valid_starts <= kept_valid_ends
    if not np.any(non_empty_mask):
        return valid_ranges
    kept_indices = np.flatnonzero(keep_mask)
    fill_indices = kept_indices[non_empty_mask]
    valid_ranges[fill_indices, 0, 0] = kept_valid_starts[non_empty_mask]
    valid_ranges[fill_indices, 1, 0] = kept_valid_ends[non_empty_mask]
    return valid_ranges

def _group_ranges_by_row(valid_ranges, step_rows, num_rows):
    step_caption_ranges_by_row = [[] for _ in range(num_rows)]
    step_caption_range_lengths_by_row = [[] for _ in range(num_rows)]
    local_step_idx_by_row = [0] * num_rows
    for flat_step_idx in range(valid_ranges.shape[0]):
        row_idx = int(step_rows[flat_step_idx, 0])
        start = int(valid_ranges[flat_step_idx, 0, 0])
        end = int(valid_ranges[flat_step_idx, 1, 0])
        local_step_idx = local_step_idx_by_row[row_idx]
        local_step_idx_by_row[row_idx] += 1
        if start == -1 and end == -1:
            interval_len = 0
        else:
            if start > end:
                raise ValueError(f'invalid range for row {row_idx}, step {local_step_idx}: start={start}, end={end}')
            interval_len = end - start + 1
        step_caption_ranges_by_row[row_idx].append([local_step_idx, start, end])
        step_caption_range_lengths_by_row[row_idx].append(interval_len)
    return (step_caption_ranges_by_row, step_caption_range_lengths_by_row)

def build_sampling_payload_from_jsonl_paths(input_paths, min_gap):
    records_by_row = []
    total_steps = 0
    max_forbidden_count = 0
    for input_path in input_paths:
        dataset_name = os.path.basename(input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'count steps {dataset_name}'):
                record = json.loads(line)
                records_by_row.append(record)
                total_steps += _count_step_bounds(record)
                max_forbidden_count = max(max_forbidden_count, _count_forbidden_intervals(record))
    num_rows = len(records_by_row)
    forbidden_capacity = max(1, max_forbidden_count)
    forbidden_starts_by_row = np.full((num_rows, forbidden_capacity), np.int32(-1), dtype=np.int32)
    forbidden_ends_by_row = np.full((num_rows, forbidden_capacity), np.int32(-1), dtype=np.int32)
    step_bounds = np.empty((total_steps, 2, 1), dtype=np.int32)
    step_rows = np.empty((total_steps, 1), dtype=np.int32)
    row_idx = 0
    write_idx = 0
    for input_path in input_paths:
        dataset_name = os.path.basename(input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            for _ in tqdm(f, desc=f'build {dataset_name}'):
                write_idx = _fill_row_arrays(records_by_row[row_idx], dataset_name, row_idx, write_idx, forbidden_starts_by_row, forbidden_ends_by_row, step_bounds, step_rows)
                row_idx += 1
    valid_ranges = _find_valid_ranges_for_all_steps(step_bounds, step_rows, forbidden_starts_by_row, forbidden_ends_by_row, min_gap)
    (step_caption_ranges_by_row, step_caption_range_lengths_by_row) = _group_ranges_by_row(valid_ranges, step_rows, num_rows)
    return {'step_caption_ranges_by_row': step_caption_ranges_by_row, 'step_caption_range_lengths_by_row': step_caption_range_lengths_by_row, 'min_gap': np.int32(min_gap)}

def build_sampling_payload(records, dataset_names, min_gap):
    if len(records) != len(dataset_names):
        raise ValueError(f'records length {len(records)} does not match dataset_names length {len(dataset_names)}')
    total_steps = 0
    max_forbidden_count = 0
    for record in records:
        total_steps += _count_step_bounds(record)
        max_forbidden_count = max(max_forbidden_count, _count_forbidden_intervals(record))
    num_rows = len(records)
    forbidden_capacity = max(1, max_forbidden_count)
    forbidden_starts_by_row = np.full((num_rows, forbidden_capacity), np.int32(-1), dtype=np.int32)
    forbidden_ends_by_row = np.full((num_rows, forbidden_capacity), np.int32(-1), dtype=np.int32)
    step_bounds = np.empty((total_steps, 2, 1), dtype=np.int32)
    step_rows = np.empty((total_steps, 1), dtype=np.int32)
    write_idx = 0
    for (row_idx, (record, dataset_name)) in enumerate(zip(records, dataset_names)):
        write_idx = _fill_row_arrays(record, dataset_name, row_idx, write_idx, forbidden_starts_by_row, forbidden_ends_by_row, step_bounds, step_rows)
    valid_ranges = _find_valid_ranges_for_all_steps(step_bounds, step_rows, forbidden_starts_by_row, forbidden_ends_by_row, min_gap)
    (step_caption_ranges_by_row, step_caption_range_lengths_by_row) = _group_ranges_by_row(valid_ranges, step_rows, num_rows)
    return {'step_caption_ranges_by_row': step_caption_ranges_by_row, 'step_caption_range_lengths_by_row': step_caption_range_lengths_by_row, 'min_gap': np.int32(min_gap)}

def build_and_save_sampling_payload(save_path, records, dataset_names, min_gap):
    payload = build_sampling_payload(records, dataset_names=dataset_names, min_gap=min_gap)
    save_sampling_payload(save_path, payload)
    return payload

def save_sampling_payload(save_path, payload):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, step_caption_ranges_by_row=np.asarray(payload['step_caption_ranges_by_row'], dtype=object), step_caption_range_lengths_by_row=np.asarray(payload['step_caption_range_lengths_by_row'], dtype=object), min_gap=payload['min_gap'])

def load_sampling_payload(load_path):
    with np.load(load_path, allow_pickle=True) as data:
        step_caption_ranges_by_row = data['step_caption_ranges_by_row'].tolist()
        step_caption_range_lengths_by_row = data['step_caption_range_lengths_by_row'].tolist()
        return {'step_caption_ranges_by_row': step_caption_ranges_by_row, 'step_caption_range_lengths_by_row': step_caption_range_lengths_by_row, 'min_gap': int(data['min_gap'])}
