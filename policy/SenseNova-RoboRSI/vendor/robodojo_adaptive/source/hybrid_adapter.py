"""Private pi0.5 proposal integration; only RGB, proprioception and robot FK."""
import inspect
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from waypoints import clip_grippers

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code/configurations/robodojo/20261005_standard_1x_10x/adapters/timeout_chunk_trials'))
from trial_adapter import Adapter as NativeAdapter
from policylab.api import InvalidAction

class Adapter(NativeAdapter):
    def __init__(self, environment_dir, options):
        super().__init__(environment_dir, options)
        if options.get('vla_horizon') not in (5,10,20,50):
            raise ValueError('VLA horizon must be 5, 10, 20, or 50')
        self.vla_horizon = options['vla_horizon']
        self.vla_port = options['vla_port']
        self.connection = None

    def vla_propose(self, subgoal):
        if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal)>2000:
            raise InvalidAction('Provide a short English subgoal')
        if self.env.end_flag[0]:
            raise InvalidAction('Episode is over')
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        if self.connection is None:
            kwargs = dict(compression=None, max_size=None, open_timeout=15)
            if 'proxy' in inspect.signature(connect).parameters:
                kwargs['proxy'] = None
            self.connection = connect(f'ws://127.0.0.1:{self.vla_port}', **kwargs)
            self.metadata = msgpack_numpy.unpackb(self.connection.recv(timeout=30))
        raw_obs = self.env.get_obs()
        state = raw_obs['state']
        names = ['left_arm_joint_state','left_ee_joint_state','right_arm_joint_state','right_ee_joint_state']
        current = np.concatenate([np.asarray(state[k]).reshape(-1) for k in names]).astype(np.float32)
        if current.shape!=(14,) or not np.isfinite(current).all():
            raise RuntimeError('Invalid public robot joint state')
        images = {}
        for output, source in [('cam_high','cam_head'),('cam_left_wrist','cam_left_wrist'),('cam_right_wrist','cam_right_wrist')]:
            rgb=np.asarray(raw_obs['vision'][source]['color'],dtype=np.uint8)
            images[output]=np.ascontiguousarray(rgb.transpose(2,0,1))
        payload=dict(state=current, images=images, prompt=subgoal)
        check=self.kinematics.check()
        start=time.monotonic()
        self.connection.send(msgpack_numpy.Packer().pack(payload))
        response=self.connection.recv(timeout=300)
        if isinstance(response,str):
            raise RuntimeError(response)
        predicted=np.asarray(msgpack_numpy.unpackb(response)['actions'],dtype=float)
        if predicted.ndim!=2 or predicted.shape[1]!=14 or len(predicted)<self.vla_horizon or not np.isfinite(predicted).all():
            raise RuntimeError(f'Invalid pi0.5 trajectory {predicted.shape}')
        raw_predicted = predicted.copy()
        predicted = clip_grippers(predicted)
        targets=[]; violations=[]
        previous={a:np.asarray(state[f'{a}_ee_pose']) for a in ['left','right']}
        for index in range(self.vla_horizon):
            dual={}; bounds=[]
            for arm,offset in [('left',0),('right',7)]:
                transform=self.kinematics.root(arm) @ self.kinematics.fk[arm].matrix(predicted[index,offset:offset+6])
                xyzw=Rotation.from_matrix(transform[:3,:3]).as_quat()
                wxyz=xyzw[[3,0,1,2]]
                pose=np.r_[transform[:3,3],wxyz]
                old=previous[arm]
                if np.linalg.norm(pose[:3]-old[:3])>.05+1e-9:
                    bounds.append(f'{arm}: displacement exceeds 5cm')
                angle=(Rotation.from_quat(wxyz[[1,2,3,0]])*Rotation.from_quat(old[[4,5,6,3]]).inv()).magnitude()
                if angle>.35+1e-9: bounds.append(f'{arm}: rotation exceeds 0.35rad')
                opening=float(predicted[index,offset+6])
                robot=self.kinematics.robots[arm]
                key=self.kinematics.manager.robot_key[self.kinematics.manager.robot_list.index(robot)]
                limits=key.data.soft_joint_pos_limits[0,robot.arm_joint_indices].cpu().numpy()
                joints=predicted[index,offset:offset+6]
                if np.any(joints<limits[:,0]) or np.any(joints>limits[:,1]): bounds.append(f'{arm}: joint range violation')
                dual[arm]=dict(position=pose[:3].tolist(),quaternion_wxyz=wxyz.tolist(),gripper_closed=bool(opening<.5))
                previous[arm]=pose
            targets.append(dual);violations.append(bounds)
        proposal=dict(proposal_id=uuid.uuid4().hex,control_step=int(self.env.take_action_cnt[0]),
                      subgoal=subgoal,targets=targets,out_of_bounds=violations,
                      selected_waypoint_indices=list(range(self.vla_horizon)),horizon=self.vla_horizon,robot_moved=False,
                      fk_validation=check,inference_latency_s=time.monotonic()-start,
                      gripper_conversion='clip opening to [0,1]; opening < 0.5 -> closed; binary current EEF contract',
                      gripper_clipped_values=int(np.count_nonzero(raw_predicted[:self.vla_horizon,[6,13]] != predicted[:self.vla_horizon,[6,13]])))
        with (self.record_dir/'vla_proposals_private.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(**proposal,raw_joint_actions=raw_predicted.tolist(),clipped_joint_actions=predicted.tolist()))+'\n')
        return proposal

    def close(self):
        if self.connection is not None:
            self.connection.close()
        super().close()
