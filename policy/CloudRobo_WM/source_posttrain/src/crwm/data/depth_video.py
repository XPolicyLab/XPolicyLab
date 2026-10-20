from __future__ import annotations
import os,logging,time as _dl_time
from pathlib import Path
import torch
import numpy as np
import av
from collections import OrderedDict
_DEPTH_DECODE_GC_INTERVAL=64
_depth_decode_counter=0
_DEPTH_DECODE_GC_INTERVAL = 64
_depth_decode_counter = 0
def _dl_timing_enabled() -> bool:
    return os.environ.get('CRWM_DL_TIMING') == '1'
def _dl_worker_id() -> int:
    """DataLoader worker id, or -1 if not in a worker process."""
    try:
        import torch.utils.data as _tud
        wi = _tud.get_worker_info()
        return int(wi.id) if wi is not None else -1
    except Exception:
        return -1
def _dl_emit(payload: dict, threshold_ms: float | None=None) -> None:
    """Print a single line `[DLTIMING] k=v k=v ...` if payload['ms'] (or
    payload['total_ms']) exceeds `threshold_ms`, OR if VERBOSE=1.
    Skips entirely when CRWM_DL_TIMING != '1'.
    """
    if not _dl_timing_enabled():
        return
    if threshold_ms is None:
        threshold_ms = float(os.environ.get('CRWM_DL_TIMING_THRESHOLD_MS', '2000'))
    verbose = os.environ.get('CRWM_DL_TIMING_VERBOSE') == '1'
    ms = payload.get('ms', payload.get('total_ms', 0))
    if not (ms > threshold_ms or verbose):
        return
    print('[DLTIMING] ' + ' '.join((f'{k}={v}' for k, v in payload.items())), flush=True)
def _decode_depth_frames_decord_x264rgb(video_path: str, timestamps: 'list[float]', tolerance_s: float, log_loaded_timestamps: bool=False) -> 'torch.Tensor':
    """Depth decode for libx264rgb dual-channel-packed videos via decord.

    Encoding (matches preprocess_depth_to_x264rgb): 3-channel rgb24,
    R = depth_hi, G = depth_lo, B = unused. Recovery: uint16 = R<<8 | G.
    decord get_batch is GIL-released and isolation-bench-faster than pyav
    on the same x264rgb file (3× cold, equal warm). Output shape matches
    pyav path: (T, H, W) uint16 mm.

    Required for any future sharded sampling: a single call can request
    `vr.get_batch(sorted_indices)` for many frames at once, batching the
    file open across N samples.
    """
    import decord
    import numpy as np
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None
    _nt = int(os.environ.get('CRWM_DECORD_THREADS', '1'))
    vr = decord.VideoReader(str(video_path), num_threads=_nt)
    average_fps = vr.get_avg_fps()
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    sorted_pos = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    sorted_indices = [frame_indices[i] for i in sorted_pos]
    frames_sorted = vr.get_batch(sorted_indices).asnumpy()
    restore = sorted(range(len(sorted_pos)), key=lambda i: sorted_pos[i])
    frames = frames_sorted[restore]
    del vr
    loaded_ts = torch.tensor([idx / average_fps for idx in frame_indices], dtype=torch.float32)
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    max_dev = (loaded_ts - query_ts).abs().max().item() if len(timestamps) else 0.0
    if max_dev >= tolerance_s:
        raise AssertionError(f'Depth (decord) timestamp alignment violation: max_dev {max_dev:.4f} > tol {tolerance_s}.\nqueried: {query_ts}\nloaded: {loaded_ts}\nvideo: {video_path}')
    if log_loaded_timestamps:
        for ts in loaded_ts.tolist():
            logging.info(f'depth(decord) frame loaded at timestamp={ts:.4f}')
    out = frames[..., 0].astype(np.uint16) << 8 | frames[..., 1].astype(np.uint16)
    result = torch.from_numpy(out)
    if _dl_t0 is not None:
        _dl_emit({'stage': 'depth_decode_decord_x264rgb', 'worker': _dl_worker_id(), 'pid': os.getpid(), 'file': Path(str(video_path)).name, 'n_ts': len(timestamps), 'shape': tuple(result.shape), 'decord_threads': _nt, 'ms': int((_dl_time.perf_counter() - _dl_t0) * 1000)}, threshold_ms=float(os.environ.get('CRWM_DL_TIMING_THRESHOLD_MS', '2000')) / 4)
    return result
def decode_depth_frames(video_path: 'Path | str', timestamps: 'list[float]', tolerance_s: float, vcodec: str, log_loaded_timestamps: bool=False) -> 'torch.Tensor':
    """Timestamp-aware lossless depth decode — mirror of decode_video_frames_torchvision.

    Given a lossless depth video (FFV1 gray16le MKV or libx264rgb packed-uint16 MP4)
    and a list of query timestamps (in seconds), return the frames at those times
    as `torch.uint16` shape `[T, H, W]` in millimetres. No normalization applied
    (raw mm). Uses pyav directly — torchcodec refuses FFV1 MKV containers and
    returns 8-bit data even for libx264rgb.

    Pattern matches decode_video_frames_torchvision so alignment with the RGB
    stream is guaranteed: both streams queried at identical timestamps, both
    with the same tolerance_s, both pick argmin over |query - loaded_pts|.
    """
    import gc as _gc
    import numpy as np
    global _depth_decode_counter
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None
    if vcodec not in ('ffv1', 'libx264rgb'):
        raise ValueError(f'Unsupported depth vcodec: {vcodec!r}')
    target_format = 'gray16le' if vcodec == 'ffv1' else 'rgb24'
    video_path = str(video_path)
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    _ts_lower_buffer_s = 0.002
    first_ts_filter = first_ts - _ts_lower_buffer_s
    if vcodec == 'libx264rgb' and os.environ.get('CRWM_DEPTH_DECORD', '1') == '1':
        return _decode_depth_frames_decord_x264rgb(video_path, timestamps, tolerance_s, log_loaded_timestamps)
    loaded_frames: list[np.ndarray] = []
    loaded_ts: list[float] = []
    container = av.open(video_path, 'r')
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_count = int(os.environ.get('CRWM_PYAV_THREADS', '2'))
            stream.thread_type = 'AUTO'
        except Exception:
            pass
        target_pts = int(first_ts / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * frame.time_base)
            if ts < first_ts_filter:
                continue
            arr = frame.to_ndarray(format=target_format)
            loaded_frames.append(arr)
            loaded_ts.append(ts)
            if log_loaded_timestamps:
                logging.info(f'depth frame loaded at timestamp={ts:.4f}')
            if ts >= last_ts:
                break
    finally:
        container.close()
        del container
        _depth_decode_counter += 1
        if _depth_decode_counter % _DEPTH_DECODE_GC_INTERVAL == 0:
            _gc.collect()
    if not loaded_frames:
        raise RuntimeError(f'No frames decoded from {video_path}')
    if vcodec == 'libx264rgb':
        loaded_frames = [f[..., 0].astype(np.uint16) << 8 | f[..., 1].astype(np.uint16) for f in loaded_frames]
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise AssertionError(f'Depth timestamp alignment violation: {min_[~is_within_tol]} > tolerance_s={tolerance_s!r}.\nqueried timestamps: {query_ts}\nloaded timestamps: {loaded_ts_t}\nvideo: {video_path}')
    frames_np = np.stack([loaded_frames[int(i)] for i in argmin_], axis=0)
    result = torch.from_numpy(frames_np)
    if _dl_t0 is not None:
        _dl_emit({'stage': 'depth_decode_pyav', 'worker': _dl_worker_id(), 'pid': os.getpid(), 'file': Path(video_path).name, 'vcodec': vcodec, 'n_ts': len(timestamps), 'shape': tuple(result.shape), 'pyav_threads': int(os.environ.get('CRWM_PYAV_THREADS', '2')), 'ms': int((_dl_time.perf_counter() - _dl_t0) * 1000)}, threshold_ms=float(os.environ.get('CRWM_DL_TIMING_THRESHOLD_MS', '2000')) / 4)
    return result
