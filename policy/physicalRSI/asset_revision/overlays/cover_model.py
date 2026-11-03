"""Shihao cover control with official RGB initial geometry and proprioception."""
import json,math,os,time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from physicalrsi_observed_geometry.robot import base_from_observation
from XPolicyLab.model_template import ModelTemplate
from .perception import initial_scene
from .curobo_planner import CuroboPlanner
ARMS=('left','right')
QUAT={'left':[.6130,-.3593,.6163,.3392],'right':[.3814,-.6197,.3785,.5720]}

class Model(ModelTemplate):
    def __init__(self,model_cfg):
        super().__init__()
        self.planners={}
        self.bases={}
        self.reset()
    def reset(self):
        for w in getattr(self,'writers',{}).values():w.release()
        if getattr(self,'trace',None):self.trace.close()
        self.planners={};self.bases={}
        self.writers={};self.trace=None;self.state=None;self.rgb=None;self.program=None;self.failure=None;self.stage='initial';self.count=0;self.targets=None;self.grip={a:1. for a in ARMS}
    def event(self,**data):
        if self.trace:self.trace.write(json.dumps({'action_index':self.count,'stage':self.stage,**data},default=lambda x:x.tolist())+'\n');self.trace.flush()
    def update_obs(self,obs):
        self.state={k:np.asarray(v).copy() for k,v in obs['state'].items() if k.endswith('_arm_joint_state') or k.endswith('_ee_joint_state') or k.endswith('_ee_pose')}
        self.rgb=np.asarray(obs['vision']['cam_head']['color']).copy()
        self.observation=obs
        if self.trace is None:
            p=Path(os.environ['COVER_TRACE_DIR'])/str(time.time_ns());p.mkdir(parents=True)
            self.trace=(p/'trace.jsonl').open('w')
            for name in ('cam_head','cam_left_wrist','cam_right_wrist'):
                rgb=np.asarray(obs['vision'][name]['color']);self.writers[name]=cv2.VideoWriter(str(p/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),25,(rgb.shape[1],rgb.shape[0]))
        for name,w in self.writers.items():w.write(cv2.cvtColor(np.asarray(obs['vision'][name]['color']),cv2.COLOR_RGB2BGR))
        if self.targets is None:self.targets={a:self.state[a+'_arm_joint_state'].astype('float32') for a in ARMS}
        self.event(state=self.state)
    def action(self, arm=None, q=None):
        return joint_action(self.targets, self.grip, ARMS, arm, q)
    def hold_gripper(self, arm, value, repeats):
        yield from hold_gripper(self.grip, arm, value, repeats, self.action)
    def move(self,arm,pose=None,max_waypoints=8):
        current=self.state[arm+'_arm_joint_state']
        planner=self.planners[arm]
        result=planner.plan_joint(current,np.zeros_like(current)) if pose is None else planner.plan_path(curr_joint_pos=current,target_ee_pose=list(map(float,pose)),real_robot_pose=self.bases[arm])
        if result.get('status')!='Success':raise RuntimeError(f'CuRobo failed {arm} {pose}')
        path=np.asarray(result['position'],np.float32);stride=max(1,math.ceil(len(path)/max_waypoints));indices=list(range(stride-1,len(path),stride))
        if not indices or indices[-1]!=len(path)-1:indices.append(len(path)-1)
        self.event(arm=arm,target_pose=pose,path_length=len(path),sampled=len(indices))
        yield from follow_waypoints(path, indices, arm, self.action)
    def observe_geometry(self):
        blocks,cups,evidence=initial_scene(self.observation)
        self.geometry_evidence=evidence
        self.cup_locations={tuple(g['pose'][:2]):g for g in evidence['cup']}
        plane=evidence['cup'][0]['support_plane']
        if not np.allclose(plane['normal'],[0,0,1],atol=.03):
            raise RuntimeError('This cover controller requires an observed horizontal support')
        table_z=-plane['offset']/plane['normal'][2]
        for arm in ARMS:
            base=base_from_observation(self.observation,arm,Path(__file__).parent/'assets/x5/X5A.urdf')
            self.bases[arm]=base
            self.planners[arm]=CuroboPlanner(robot_origin_pose=[0,0,0,1,0,0,0],active_joints_name=[f'joint{i}' for i in range(1,7)],all_joints=[f'joint{i}' for i in range(1,7)],dt=.004,yml_path=str(Path(__file__).parent/'assets/x5/curobo.yml'),table_height=table_z-base[2])
        self.table_z=table_z
        return blocks,cups,evidence

    def transfer(self,source,target,arm):
        geometry=self.cup_locations[tuple(source)]
        height=geometry['dims'][2]
        # Fixed tool transform describes the robot, not an environment object.
        quat=QUAT[arm];rotation=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        tool_offset=-.15757000232458113*rotation[:,0]
        center_z=self.table_z+height/2
        contact=center_z+tool_offset[2]
        source_tool=np.asarray(source)+tool_offset[:2];target_tool=np.asarray(target)+tool_offset[:2]
        parameters=PickPlaceParameters(approach_height=contact+.07,contact_height=contact,
            lift_height=contact+.08,release_height=contact+.005,retract_height=contact+.07,
            open_grip=1.,grasp_grip=.54,prepare_steps=2,grasp_steps=6,release_steps=6,
            contact_waypoints=5,lift_waypoints=5,release_waypoints=5,retract_waypoints=5,home_waypoints=10)
        yield from pick_place(tuple(source_tool),tuple(target_tool),arm,quat,parameters,
            move=self.move,hold_gripper=self.hold_gripper)
        self.cup_locations.pop(tuple(source))
        self.cup_locations[tuple(target)]=geometry
    def run(self):
        blocks,cups,evidence=self.observe_geometry();order=sorted(blocks,key=lambda k:blocks[k][0]);homes=dict(zip(order,cups,strict=True));self.event(initial_geometry=evidence,color_order=order)
        for color in order:
            self.stage='cover_'+color;source=homes[color];yield from self.transfer(source,blocks[color],'left' if source[0]<=.05 else 'right');self.event(program_transfer_finished=True,attachment='unknown')
        for color in ('red','green','blue'):
            self.stage='uncover_'+color;source=blocks[color];yield from self.transfer(source,homes[color],'left' if source[0]<=.05 else 'right');self.event(program_transfer_finished=True,attachment='unknown')
        self.stage='home'
        for arm in ARMS:yield from self.move(arm,max_waypoints=10)
        self.stage='program_finished_hold';self.event(official_score=None)
    def get_action(self):
        if self.program is None and self.failure is None:self.program=self.run()
        try:action=next(self.program) if self.failure is None else self.action()
        except StopIteration:action=self.action()
        except Exception as exc:
            self.failure=repr(exc);self.stage='failure_hold';self.targets={a:self.state[a+'_arm_joint_state'].astype('float32') for a in ARMS};self.grip={a:float(self.state[a+'_ee_joint_state'][0]) for a in ARMS};self.event(failure=self.failure);action=self.action()
        self.count+=1;self.event(action=action)
        return [action]


# Local extraction prototype; the original source remains unchanged.
from physicalrsi_skill_library.manipulation_programs import execute_memory
class StructuredModel(Model):
    _operation_revision = 'f23d926176259c174cd4c60e16290bd4c05189e1874c8aee00e0c915c539f62c'
    _operation_bindings = [('motion.bind', '_memory_motion_bind'), ('motion.advance', '_memory_motion_advance'), ('motion.record', '_memory_motion_record')]
    _extra_bindings = [('vision.geometry','_geometry')]
    _memory_path = str(__import__('pathlib').Path(__file__).with_name('structured_memory.json'))
    def _memory_motion_bind(self, packet, context):
        if self.program is None and self.failure is None:
            self.program = self.run()
        return None
    def _memory_motion_advance(self, packet, context):
        try:
            action = next(self.program) if self.failure is None else self.action()
        except StopIteration:
            action = self.action()
        except Exception as exc:
            self.failure = repr(exc)
            self.stage = 'failure_hold'
            self.targets = {a: self.state[a + '_arm_joint_state'].astype('float32') for a in ARMS}
            self.grip = {a: float(self.state[a + '_ee_joint_state'][0]) for a in ARMS}
            self.event(failure=self.failure)
            action = self.action()
        return action
    def _memory_motion_record(self, packet, context):
        action=packet
        self.count += 1
        self.event(action=action)
        return [action]
    def get_action(self):
        return execute_memory(self, 'action', None)
    def _geometry(self, packet, context):
        blocks, cups, evidence = self.observe_geometry()
        order = sorted(blocks, key=lambda k: blocks[k][0])
        homes = dict(zip(order, cups, strict=True))
        self.event(initial_geometry=evidence, color_order=order)
        return {'blocks':blocks,'cups':cups,'order':order,'homes':homes}
    def run(self):
        geometry=execute_memory(self,'geometry',None)
        blocks,cups,order,homes=(geometry[k] for k in ('blocks','cups','order','homes'))
        for color in order:
            self.stage = 'cover_' + color
            source = homes[color]
            yield from self.transfer(source, blocks[color], 'left' if source[0] <= 0.05 else 'right')
            self.event(program_transfer_finished=True, attachment='unknown')
        for color in ('red', 'green', 'blue'):
            self.stage = 'uncover_' + color
            source = blocks[color]
            yield from self.transfer(source, homes[color], 'left' if source[0] <= 0.05 else 'right')
            self.event(program_transfer_finished=True, attachment='unknown')
        self.stage = 'home'
        for arm in ARMS:
            yield from self.move(arm, max_waypoints=10)
        self.stage = 'program_finished_hold'
        self.event(official_score=None)
Model = StructuredModel

from physicalrsi_skill_library.joint_skills import joint_action, hold_gripper, follow_waypoints

from physicalrsi_skill_library.pick_place import PickPlaceParameters, pick_place
