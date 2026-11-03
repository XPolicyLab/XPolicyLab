"""Independent candidate: observed button anchors, unchanged press controller."""
import json
import time
from pathlib import Path
import numpy as np
from rdj_rgb_adapters.press_by_number import Model as IncumbentModel
from rdj_master.tasks.press_by_number import cap as baseline
from .press_perception import PressPerception
from physicalrsi_observed_geometry import GeometryUnavailable


# A bounded exploration distance, not an assumed physical button travel.
PROBE_BUDGET_M = .04


def observed_anchors(positions):
    anchors={}
    positions=np.asarray(positions,dtype=float)
    if positions.shape!=(3,3) or not np.isfinite(positions).all():
        raise ValueError('Three current observed button positions required')
    minimum_transit=float(positions[:,2].max())+baseline.PRESS_TOOL_TIP_OFFSET_M+baseline.RELEASE_CLEARANCE_M
    for label,xyz in zip(baseline.BUTTON_LABELS,positions,strict=True):
        tip_pose=(*xyz,1.,0.,0.,0.);axis=(0.,0.,1.)
        up=baseline.tool_tip_pose_to_ee_pose(baseline.make_up_tip_pose(tip_pose,axis))
        probe_tip=list(tip_pose);probe_tip[2]-=PROBE_BUDGET_M
        probe=baseline.tool_tip_pose_to_ee_pose(probe_tip)
        anchors[label]=baseline.ButtonAnchor(tip_pose=tip_pose,axis_world=axis,released_ratio=1.,
            joint_travel_m=PROBE_BUDGET_M, # Legacy container field; exploration limit only.
            approach_pose=up,down_pose=probe,released_surface_pose=baseline.tool_tip_pose_to_ee_pose(tip_pose),
            up_pose=up,high_pose=baseline.make_high_transit_ee_pose(up,minimum_z_m=minimum_transit),retry_down_pose=probe)
    return anchors

class ObservedCapPolicy(baseline.CapPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.perception=PressPerception(cfg['perception'],baseline.NumberCardReader._classify)
        self.output=Path(cfg['evidence']);self.output.mkdir(parents=True,exist_ok=True)
        self.observation_number=0

    def _observe(self, env_idx, observation):
        if env_idx in self._slots:
            return super()._observe(env_idx,observation)
        rgb=np.asarray(observation['vision']['cam_head']['color'])
        measured=self.perception.observe(observation)
        robot=baseline._robot_state(observation)
        memory=baseline.PressPolicyAPI.reset(*measured['digits'])
        anchors=observed_anchors(measured['positions'])
        session=baseline.PressSkillSession();goal=self._policy.decide(memory)
        session.start(self._skills.compile(goal,anchors,robot),goal,anchors,robot,return_home_on_release=baseline._return_home_on_release(memory,goal))
        self._slots[env_idx]=baseline._EnvSlot(memory,anchors,dict(robot),dict(robot),session)
        (self.output/f'localization-{self.observation_number:06d}.json').write_text(json.dumps(measured,indent=2)+'\n')
        self.observation_number+=1

class Model(IncumbentModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._core=ObservedCapPolicy(model_cfg)

    def update_obs(self, obs):
        return self.update_obs_batch([dict(obs, env_idx=0)])

    def update_obs_batch(self, obs_list):
        waiting = getattr(self, '_localization_wait', {})
        self._localization_wait = waiting
        self._observed_indices = tuple(int(obs['env_idx']) for obs in obs_list)
        if not self._observed_indices or len(set(self._observed_indices)) != len(self._observed_indices):
            raise ValueError('A nonempty batch with unique env_idx values is required')
        for obs in obs_list:
            index = int(obs['env_idx'])
            previous = waiting.get(index)
            robot = baseline._robot_state(obs)
            if previous is not None:
                previous['robot'] = robot
                previous['observations'] += 1
                if previous['observations'] % 16:
                    continue
            try:
                # A missing observation in one environment must not abort peers.
                self._core.update_obs_batch([obs])
            except GeometryUnavailable as error:
                if index in self._core._slots:
                    raise
                if previous is None:
                    arrays = {f'vision/{name}/{key}': np.asarray(camera[key])
                              for name, camera in obs['vision'].items()
                              for key in ('color', 'intrinsic_matrix', 'extrinsic_matrix')}
                    arrays.update({f'state/{key}': np.asarray(value)
                                   for key, value in robot.items()})
                    np.savez_compressed(self._core.output / f'unobserved-{index}-{time.time_ns()}.npz', **arrays)
                    waiting[index] = {'robot': robot, 'observations': 0}
                print(f'PRESS_GEOMETRY_WAIT env={index} reason={error}', flush=True)
            else:
                waiting.pop(index, None)

    def get_action_batch(self, env_idx_list=None):
        indices = self._observed_indices if env_idx_list is None else tuple(map(int, env_idx_list))
        ready = [index for index in indices if index in self._core._slots]
        active = dict(zip(ready, super().get_action_batch(ready), strict=True)) if ready else {}
        return [active[index] if index in active else
                [{key: np.asarray(value).copy() for key, value in self._localization_wait[index]['robot'].items()}]
                for index in indices]

    def get_action(self):
        return self.get_action_batch([0])[0]

    def reset(self):
        self._localization_wait = {}
        self._observed_indices = ()
        return super().reset()

    def prepare_case(self, case_meta=None):
        return self.reset()

    def on_trial_end(self, result=None):
        return self.reset()

    def close(self):
        self._core.perception.close()
