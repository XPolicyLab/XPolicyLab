"""Read only video/history frames needed by selected temporal RoboDojo samples."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from worldscape_policy.data.adapters.common import (
    LanguageTemporalPacker, _history_timeline_indices, _optional_dependency,
    _temporal_labels,
)
from worldscape_policy.data.adapters.robodojo import (
    _CAMERA_PATHS, _required_dataset,
)
from worldscape_policy.training.timing_trace import TraceSpan

if TYPE_CHECKING:
    from worldscape_policy.data.adapters.robodojo import RoboDojoJointHDF5Dataset

EpisodePayload = dict[str, Any]
SelectedPayload = dict[int, EpisodePayload]


def load_selected_frames(
    dataset: RoboDojoJointHDF5Dataset,
    indices: Iterable[int],
    *,
    cached: Mapping[int, EpisodePayload] | None = None,
) -> SelectedPayload:
    """Return immutable sparse payloads; states/actions remain full episodes.

    Sparse camera arrays retain the original timeline length. Unread elements
    are None and fail loudly if sample construction ever requests an unplanned
    frame. Returned arrays never mutate a payload held by a sample thread.
    """
    if (not dataset.temporal_packing or dataset.robodojo_visual_prompt != 'none'
            or dataset.cache_video_decode):
        raise ValueError('selected frames require temporal RoboDojo text with compressed caching')
    cached = {} if cached is None else cached
    episodes = defaultdict(list)
    for index in indices:
        episode, anchor = dataset.index[index]
        episodes[int(episode)].append(int(anchor))
    result = {}
    h5py = _optional_dependency('h5py', 'train')
    for episode, anchors in episodes.items():
        arrays = cached.get(episode)
        if arrays is None:
            with TraceSpan('joint_metadata_read', path=str(dataset.paths[episode])):
                arrays = dataset._read_arrays(episode, load_images=False)
        length = len(arrays['actions'])
        labels, segments = _temporal_labels(arrays, length, None)
        videos, heads = set(), set()
        packer = LanguageTemporalPacker(
            max_chunk_size=dataset._episode_max_chunk_size(episode),
            chunk_size=dataset.action_horizon)
        for anchor in anchors:
            packed = packer.indices(anchor, labels, trajectory_ids=(
                segments if dataset.respect_trajectory_segments else None))
            videos.update(map(int, packed.video))
            if dataset.history_sampler.num_frames > 0:
                heads.update(map(int, _history_timeline_indices(
                    history_sampler=dataset.history_sampler,
                    history_window=dataset.history_window,
                    anchor_index=int(packed.anchors[0]), trajectory_length=length,
                    use_native_history_sampler=False, embodiment=dataset.embodiment)))
        missing = {}
        requested_count = 0
        for camera in _CAMERA_PATHS:
            requested = videos | heads if camera.endswith('.head') else videos
            requested_count += len(requested)
            positions = sorted(i for i in requested if arrays[camera][i] is None)
            if positions:
                missing[camera] = positions
        with TraceSpan('frame_coverage', path=str(dataset.paths[episode]),
                       requested_frames=requested_count,
                       missing_frames=sum(map(len, missing.values())),
                       cached_frames=requested_count - sum(map(len, missing.values()))):
            pass
        if missing:
            arrays = dict(arrays)
            with TraceSpan('hdf5_episode', path=str(dataset.paths[episode]),
                           episode=episode):
                with TraceSpan('hdf5_open'):
                    handle = h5py.File(dataset.paths[episode], 'r')
                with handle:
                    _fill_cameras(handle, arrays, missing)
        result[episode] = arrays
    return result


def _fill_cameras(handle, arrays, missing):
    """Use one HDF5 selection for encoded frames, without loading their gaps."""
    for camera, positions in missing.items():
        source = _required_dataset(handle, _CAMERA_PATHS[camera])
        start, stop = positions[0], positions[-1] + 1
        # Raw video retains its bounded contiguous read. For JPEG/PNG datasets,
        # sorted unique indices let HDF5 gather exactly the requested entries.
        indexed = source.ndim == 1 and len(positions) < stop - start
        with TraceSpan('camera_read', camera=camera, requested_frames=len(positions),
                       span_frames=stop - start, selection='indexed' if indexed else 'span',
                       chunks=source.chunks, compression=source.compression,
                       dtype=str(source.dtype)) as trace:
            payload = source[positions if indexed else slice(start, stop)]
            if trace.active:
                returned_bytes = (sum(value.nbytes if isinstance(value, np.ndarray) else len(value)
                                      for value in payload) if payload.dtype.kind == 'O'
                                  else payload.nbytes)
                trace.add(returned_bytes=returned_bytes, read_frames=len(payload))
            values = payload if indexed else [payload[position - start] for position in positions]
        sparse = arrays[camera].copy()
        for position, value in zip(positions, values, strict=True):
            # Own raw frame memory; never retain an entire temporary range.
            sparse[position] = value.copy() if isinstance(value, np.ndarray) else value
        arrays[camera] = sparse
