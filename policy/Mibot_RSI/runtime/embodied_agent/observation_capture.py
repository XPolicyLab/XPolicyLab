"""Fixed-state RGB capture for RoboDojo's render/annotator pipeline.

Keep the empirically tested six render updates, but only read/record the final
observation. This is a conservative pipeline workaround, not a sensor timestamp
or a claim that six updates are universally necessary.
"""
import time


def _clock(env):
    sim = env.sim.sim
    return (int(env.take_action_cnt[0]), int(sim.current_time_step_index), float(sim.current_time))


def capture_observation(env):
    before = _clock(env)
    started = time.perf_counter()
    # get_obs itself renders once and appends a frame to the upstream video.
    # Warm up only the renderer; do not duplicate those read/record side effects.
    for _ in range(5):
        env.render()
    refreshed = time.perf_counter()
    obs = env.get_obs()
    finished = time.perf_counter()
    after = _clock(env)
    if before != after:
        raise RuntimeError('Observation capture advanced the simulation clock; frame rejected')
    return obs, {
        'mode': 'six_renders_one_observation_read',
        'render_passes': 6,
        'observation_reads': 1,
        'warmup_render_wall_s': refreshed - started,
        'final_get_obs_wall_s': finished - refreshed,
        'total_wall_s': finished - started,
        'clock_before': {'control_step': before[0], 'physics_step': before[1], 'simulation_time_s': before[2]},
        'clock_after': {'control_step': after[0], 'physics_step': after[1], 'simulation_time_s': after[2]},
        'scope': 'Empirical fixed-state renderer refresh; no sensor timestamp guarantee',
    }
