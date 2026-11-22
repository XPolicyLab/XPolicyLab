"""Server-only exact optimizations for the deployed history5 runtime.

Adapted from colin's 8ba3d70 exact uint8 lookup and value-keyed time caching.
Keep THIS deployment's VAPv2 CPU float32 division followed by GPU scaling,
history Memory, FP32 head/accumulation, original attention and model weights.
No compact-F3 model replacement. A runtime instance belongs to one worker.
"""
from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import torch


def install(runtime, *, pixel=True, time_cache=True, verify_actions=True):
    if not (pixel or time_cache):
        return dict(enabled=False, mode='off')
    import wall_x.model.vapv2.inference_acceleration as acceleration
    output = Path(os.environ['VAP_RUN_DIR']) / 'exact_acceleration.json'
    state = dict(enabled=True, mode='both' if pixel and time_cache else 'pixel' if pixel else 'time', failed=False, verify_latents=False, verify_time=False, pixel_checks=[], action_checks=[], memo_hits=0,
                 memo_misses=0, reference='8ba3d70bc7f2fc2291d8e8dae63a9147d54467c1',
                 adaptations=['VAPv2 CPU float32 /255 then GPU multiply/subtract normalization',
                              'history5-compatible action time cache; original full shape'],
                 retained=['history5', 'fp32_head_accum', 'graph_rope', 'original client and IK'])
    active = [True]
    lookup = {}
    memo = OrderedDict()
    original_video = runtime.encode_video
    original_pair = acceleration._time_pair
    checked = set()

    def save():
        state['memo_entries'] = len(memo)
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(output)

    @torch.no_grad()
    def pixels(p):
        low = False  # VAPv2 explicitly normalizes all uint8 input, including 0/1.
        key = (str(runtime.device), runtime.dtype, low)
        if key not in lookup:
            values = (torch.arange(256, device='cpu', dtype=torch.uint8).float() / 255).to(runtime.device)
            lookup[key] = (values * 2.0 - 1.0).to(runtime.dtype)
        return lookup[key][p.to(runtime.device).long()].permute(0, 4, 1, 2, 3)

    @torch.no_grad()
    def encode(video):
        p = torch.as_tensor(video)
        if (not active[0] or state['mode'] not in ('pixel','both') or p.device.type != 'cpu' or p.dtype != torch.uint8
                or not p.is_contiguous() or torch.device(runtime.device).type != 'cuda'
                or runtime.dtype != torch.bfloat16):
            return original_video(video)
        if p.ndim != 5 or tuple(p.shape[-3:]) != (int(runtime.cfg.model.height),int(runtime.cfg.model.width),3) or p.shape[0] != len(runtime.view_ids):
            return original_video(video)
        frames = p.shape[1]
        if frames not in (1, 1 + runtime.video_frames_per_chunk) or (frames-1) % runtime.vae_temporal_stride:
            return original_video(video)
        flat = pixels(p)
        key = (tuple(p.shape), bool(p.max().item() <= 1))
        if key not in checked:
            ref = (p.float() / 255).to(device=runtime.device)
            ref = ref * 2.0 - 1.0
            ref = ref.to(runtime.dtype).permute(0, 4, 1, 2, 3)
            assert torch.equal(flat, ref), 'Pixel lookup changed VAE inputs'
            assert flat.stride() == ref.stride(), 'Pixel layout changed'
            checked.add(key)
            state['pixel_checks'].append(dict(shape=list(p.shape), low_range=key[1], bitwise_equal=True))
            save()
        latents = runtime.model.vae.encode(flat, device=runtime.device,
                                          tiled=runtime.vae_tiled).to(device=runtime.device, dtype=runtime.dtype)
        if latents.shape[2] != (frames-1)//runtime.vae_temporal_stride+1:
            raise RuntimeError('VAE temporal shape changed')
        if state['verify_latents']:
            reference=original_video(video)
            state.setdefault('latent_checks',[]).append(dict(shape=list(p.shape), equal=torch.equal(reference,latents.unsqueeze(0)), max_abs=float((reference-latents.unsqueeze(0)).abs().max())))
            save()
        return latents.unsqueeze(0)

    @torch.no_grad()
    def pair(core, x, timestep, cache):
        if not active[0] or state['mode'] not in ('time','both'):
            return original_pair(core, x, timestep, cache)
        if torch.is_grad_enabled() or core.training:
            raise RuntimeError('Time caching requires eval/no_grad')
        # Only action-time parameters affect this tensor; prefix contents do not.
        signature = tuple((p.data_ptr(), p._version, p.dtype, p.device)
                          for p in core.action_time.parameters())
        key = (id(core), signature, tuple(x.shape), x.dtype, x.device,
               tuple(timestep.shape), timestep.dtype,
               tuple(timestep.detach().cpu().reshape(-1).tolist()),
               tuple(cache.full_positions.shape), cache.active_start)
        if key not in memo:
            value = original_pair(core, x, timestep, cache)
            if len(memo) >= 64:
                memo.popitem(last=False)
            memo[key] = value
            state['memo_misses'] += 1
        else:
            state['memo_hits'] += 1
            memo.move_to_end(key)
        if state['verify_time']:
            reference=original_pair(core,x,timestep,cache)
            checks=[dict(equal=torch.equal(a,b), max_abs=float((a-b).abs().max())) for a,b in zip(reference,memo[key])]
            state.setdefault('time_checks',[]).append(checks)
            save()
        return memo[key]

    if pixel:
        runtime.encode_video = encode
    if time_cache:
        acceleration._time_pair = pair
    for name in (('sample_first_action', 'sample_next_action') if verify_actions else ()):
        original = getattr(runtime, name)
        def wrap(*args, _original=original, _name=name, **kwargs):
            if state['failed']:
                raise RuntimeError('Acceleration equality gate already failed; refuse further actions')
            # Compare complete actions with original inputs, history and seed.
            if not any(c['method'] == _name for c in state['action_checks']):
                start = time.monotonic()
                try:
                    active[0] = False
                    with torch.random.fork_rng(devices=[torch.device(runtime.device).index or 0]):
                        reference = _original(*args, **kwargs)
                finally:
                    active[0] = True
                candidate = _original(*args, **kwargs)
                equal = torch.equal(reference, candidate)
                state['action_checks'].append(dict(method=_name, bitwise_equal=equal,
                    max_abs=float((reference-candidate).abs().max()), seed=int(kwargs['seed']),
                    steps=int(runtime.cfg.model.action_flow['inference_steps']),
                    seconds=time.monotonic()-start))
                save()
                print('EXACT_ACCELERATION_ACTION_CHECK', json.dumps(state['action_checks'][-1]), flush=True)
                if not equal:
                    state['failed']=True
                    save()
                    raise RuntimeError('Server acceleration changed complete action output')
                return candidate
            result = _original(*args, **kwargs)
            save()
            return result
        setattr(runtime, name, wrap)
    # Exhaustive values, including VAPv2's explicit normalization of near-black input.
    for vals in ((torch.arange(256, dtype=torch.uint8), torch.tensor([0,1], dtype=torch.uint8)) if pixel else ()):
        p = vals.repeat(3).reshape(1, 1, 1, -1, 3)
        ref = (p.float() / 255).to(device=runtime.device)
        ref = (ref * 2.0 - 1.0).to(runtime.dtype)
        assert torch.equal(pixels(p), ref.permute(0,4,1,2,3))
    state['all_256_pixel_values_and_low_range_verified'] = bool(pixel)
    save()
    return state
