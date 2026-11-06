"""Evaluator recording; the skill context receives only PublicRecorder."""
from pathlib import Path
import json
import numpy as np
from .clock import OFFICIAL_ACTION_KEYS, POLICY_ACTION_FORMAT

def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

class NativeRecorder:

    def __init__(self, env, output, args, robots, *, task_name, public_recorder_class):
        self.env = env
        self.output_root = Path(output)
        self.args = args
        self.task_name = task_name
        self.public = public_recorder_class(self.output_root)
        self.robots = robots
        self._sim_step = 0
        self.phases = []
        self.sim_steps = []
        self.commands = []
        self.camera_calibration = {}

    def record(self, command=None, phase=''):
        self.env.obs_manager.render_for_capture()
        frame = self.env.obs_manager.get_obs(env_idx_list=[0])[0]
        self.public.append(frame, self.robots)
        for camera, item in frame.get('vision', {}).items():
            if camera not in self.camera_calibration:
                self.camera_calibration[camera] = {'intrinsic_matrix': jsonable(item.get('intrinsic_matrix')), 'extrinsic_matrix': jsonable(item.get('extrinsic_matrix'))}
        self.phases.append(str(phase))
        self.sim_steps.append(self._sim_step)
        self.commands.append(jsonable(command or {}))

    def save(self, episode_index, success, score, instruction, extra):
        import h5py
        directory = self.output_root / f'episode_{episode_index:07d}'
        directory.mkdir()
        videos = {}
        if not self.args.no_video:
            import imageio.v2 as imageio
            for camera, frames in self.public.images.items():
                name = f'video_{camera}.mp4'
                with imageio.get_writer(directory / name, fps=25, codec='libx264', quality=8, macro_block_size=1) as writer:
                    for frame in frames:
                        writer.append_data(frame[..., :3])
                videos[camera] = name
        metadata = jsonable({'task': self.task_name, 'seed': self.args.seed, 'episode_index': episode_index, 'success': success, 'score': score, 'instruction': instruction, 'frame_count': len(self.phases), 'frame_interval_sim_steps': int(self.env.obs_manager.collect_interval), 'phase': self.phases, 'camera_names': sorted(self.public.images), 'video_files': videos, 'image_files': {}, 'images_in_hdf5': self.args.images_in_hdf5, 'gripper_observation_encoding': 'normalized_singleton_not_measured_fingers', **extra})
        with h5py.File(directory / 'trajectory.hdf5', 'w') as handle:
            handle.attrs['num_actions'] = len(self.phases) - 1
            handle.attrs['num_observations'] = len(self.phases)
            handle.attrs['metadata_json'] = json.dumps(metadata, ensure_ascii=False)
            handle.attrs['complete'] = True
            observations = handle.create_group('observations')
            observations.create_dataset('sim_step', data=np.asarray(self.sim_steps, dtype=np.int64))
            states = observations.create_group('state')
            for prefix in ('left', 'right'):
                states.create_dataset(prefix + '_arm_joint_state', data=getattr(self.public, 'state_' + prefix + '_arm'))
                states.create_dataset(prefix + '_ee_pose', data=getattr(self.public, 'state_' + prefix + '_ee'))
                states.create_dataset(prefix + '_ee_joint_state', data=self.public.public_grippers[prefix])
            if self.args.images_in_hdf5:
                images = observations.create_group('images')
                for camera, frames in self.public.images.items():
                    images.create_dataset(camera, data=np.asarray(frames, dtype=np.uint8), compression='gzip', shuffle=True)
            actions = handle.create_group('actions')
            actions.create_dataset('applied_json', data=[json.dumps(item) for item in self.commands[1:]], dtype=h5py.string_dtype('utf-8'))
            timestamps = handle.create_group('timestamps')
            timestamps.create_dataset('action_sim_step', data=np.asarray(self.sim_steps[1:], dtype=np.int64))
        (directory / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        return directory
