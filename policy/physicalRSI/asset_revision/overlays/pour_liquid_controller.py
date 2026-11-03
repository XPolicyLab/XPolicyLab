from pathlib import Path
import os,json,time
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from XPolicyLab.model_template import ModelTemplate
from .controller import PrivilegedPourLiquidIntoCupModel
from .plan import resolve_pour_plan
from .perception import registration,pose,observed_bottle_fit
from physicalrsi_observed_geometry.calibrated_depth_scene import calibrated_depth_scene
from itertools import product
from .sam.sam3_http import segment_many_http

class Model(ModelTemplate):
    def __init__(self,model_cfg):
        super().__init__()
        self.core=PrivilegedPourLiquidIntoCupModel(model_cfg)
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.plans={};self.ticks={}
        self.out=Path(os.environ['POUR_TRACE']);self.out.mkdir(parents=True,exist_ok=True)
    def reset(self):
        self.core.reset();self.plans.clear();self.ticks.clear()
    def initial(self,o):
        i=o['env_idx']; rgb=np.asarray(o['vision']['cam_head']['color'],dtype=np.uint8)[:,:,:3]
        cv2.imwrite(str(self.out/f'env{i}_initial.png'),rgb[:,:,::-1])
        groups=segment_many_http(rgb,[('bottle',.5),('cup',.5)],timeout_s=600)
        inspection=getattr(self,'inspections',{}).get(i)
        geometry={}; fits={};scene=inspection.scene if inspection is not None else calibrated_depth_scene(o)
        for label,g in zip(['bottle','cup'],groups):
            results=[]
            for j,mask in enumerate(g.masks):
                cv2.imwrite(str(self.out/f'env{i}_{label}_{j}.png'),mask.astype('uint8')*255)
                if mask.sum()<80:continue
                results.append(observed_bottle_fit(mask,scene) if label=='bottle' else registration(mask,label,scene=scene))
            if not results:raise RuntimeError(f'{label} no RGB masks')
            fit=min(results,key=lambda f:f['error_px'])
            fits[label]=fit
            if fit['error_px']>4 or fit['iou']<.45:raise RuntimeError(f'{label} fit rejected {fit}')
            geometry[label+'_pose']=pose(fit['pose']).tolist()
            points=fit['points_local'];low,high=np.quantile(points,[.01,.99],axis=0)
            geometry[label+'_vertices']=[list(p) for p in product(*zip(low,high))]
        shape=np.asarray(fits['cup']['points_local']);top=shape[shape[:,2]>=shape[:,2].max()-.010]
        center=(top[:,:2].min(0)+top[:,:2].max(0))/2
        inner_rim_radius=float(np.quantile(np.linalg.norm(top[:,:2]-center,axis=1),.10))
        geometry['vessel_radius']=inner_rim_radius
        geometry['inner_rim_radius']=inner_rim_radius
        geometry['cup_outer_radius']=float(np.max(np.linalg.norm(top[:,:2]-center,axis=1)))
        geometry['cup_center_local']=[*center.tolist(),float(shape[:,2].max())]
        bottle=np.asarray(fits['bottle']['points_local']);rim=bottle[bottle[:,2]>=bottle[:,2].max()-.010]
        rim_center=(rim[:,:2].min(0)+rim[:,:2].max(0))/2
        geometry['spout_radius']=float(np.quantile(np.linalg.norm(rim[:,:2]-rim_center,axis=1),.1))
        geometry['robot_config']={'gripper_bias':.145,'inv_delta_matrix':np.eye(3).tolist()}
        if inspection is not None:
            from .perception import bind_liquid_opening
            bind_liquid_opening(geometry,inspection.opening)
        (self.out/f'env{i}_geometry.json').write_text(json.dumps({'fits':fits,'geometry':geometry,'depth_calibration':scene.depth_calibration_report},default=lambda x:x.tolist(),indent=2))
        return resolve_pour_plan(geometry,i,o['state'])
    def update_obs_batch(self,obs):
        pending=[o for o in obs if o['env_idx'] not in self.plans]
        futures=[(o,self.pool.submit(self.initial,o)) for o in pending]
        for o,f in futures:
            i=o['env_idx']
            try:self.plans[i]=f.result()
            except Exception as e:
                print(json.dumps({'event':'perception_failed','env':i,'error':str(e)}),flush=True)
                self.plans[i]={'skill_choice':'infeasible','arm':'none','stages':[]}
        envs=[]
        for o in obs:
            i=o['env_idx'];t=self.ticks.get(i,0);self.ticks[i]=t+1
            if t%20==0:
                rgb=np.asarray(o['vision']['cam_head']['color'],dtype=np.uint8)[:,:,:3]
                cv2.imwrite(str(self.out/f'env{i}_a{t:03d}.jpg'),rgb[:,:,::-1])
                for cam in ['cam_left_wrist','cam_right_wrist']:
                    im=np.asarray(o['vision'][cam]['color'],dtype=np.uint8)[:,:,:3]
                    cv2.imwrite(str(self.out/f'env{i}_{cam}_a{t:03d}.jpg'),im[:,:,::-1])
            plan=self.plans[i]
            envs.append({'env_idx':i,'robot':{k:o['state'][k] for k in ['left_ee_pose','right_ee_pose','left_ee_joint_state','right_ee_joint_state']},'skill_choice':plan['skill_choice'],'arm':plan['arm'],'stages':plan['stages']})
        self.core.update_obs_batch(envs)
    def get_action_batch(self,ids=None):return self.core.get_action_batch(ids)
    def update_obs(self,o):self.update_obs_batch([o])
    def get_action(self):return self.get_action_batch([0])[0]


# Local extraction prototype; the original source remains unchanged.
from physicalrsi_skill_library.manipulation_programs import execute_memory
class StructuredModel(Model):
    _operation_revision = '9408c6826bd5ba2653f61e3862cc67b02952a8abe9e310e20526dc8c98614e80'
    _operation_bindings = [('vision.plan', '_memory_vision_plan'), ('motion.envelopes', '_memory_motion_envelopes'), ('motion.bind', '_memory_motion_bind')]
    _memory_path = str(__import__('pathlib').Path(__file__).with_name('structured_memory.json'))
    def _memory_vision_plan(self, packet, context):
        obs=packet
        pending = [o for o in obs if o['env_idx'] not in self.plans]
        futures = [(o, self.pool.submit(self.initial, o)) for o in pending]
        for o, f in futures:
            i = o['env_idx']
            try:
                self.plans[i] = f.result()
            except Exception as e:
                print(json.dumps({'event': 'perception_failed', 'env': i, 'error': str(e)}), flush=True)
                self.plans[i] = {'skill_choice': 'infeasible', 'arm': 'none', 'stages': []}
        return obs
    def _memory_motion_envelopes(self, packet, context):
        obs=packet
        envs = []
        for o in obs:
            i = o['env_idx']
            t = self.ticks.get(i, 0)
            self.ticks[i] = t + 1
            if t % 20 == 0:
                rgb = np.asarray(o['vision']['cam_head']['color'], dtype=np.uint8)[:, :, :3]
                cv2.imwrite(str(self.out / f'env{i}_a{t:03d}.jpg'), rgb[:, :, ::-1])
                for cam in ['cam_left_wrist', 'cam_right_wrist']:
                    im = np.asarray(o['vision'][cam]['color'], dtype=np.uint8)[:, :, :3]
                    cv2.imwrite(str(self.out / f'env{i}_{cam}_a{t:03d}.jpg'), im[:, :, ::-1])
            plan = self.plans[i]
            envs.append({'env_idx': i, 'robot': {k: o['state'][k] for k in ['left_ee_pose', 'right_ee_pose', 'left_ee_joint_state', 'right_ee_joint_state']}, 'skill_choice': plan['skill_choice'], 'arm': plan['arm'], 'stages': plan['stages']})
        return envs
    def _memory_motion_bind(self, packet, context):
        envs=packet
        self.core.update_obs_batch(envs)
    def update_obs_batch(self,obs):
        return execute_memory(self,'action',obs)

Model = StructuredModel


class ObservedCupModel(StructuredModel):
    def __init__(self,model_cfg):
        super().__init__(model_cfg)
        self.inspections={};self.inspection_actions={};self.latest_ids=[]
    def reset(self):
        super().reset()
        self.inspections={};self.inspection_actions={};self.latest_ids=[]
    def update_obs_batch(self,obs):
        from .cup_inspection import CupInspection
        self.latest_ids=[o['env_idx'] for o in obs]
        self.inspection_actions={};ready=[]
        for o in obs:
            i=o['env_idx']
            if i not in self.inspections:self.inspections[i]=CupInspection(self.out/f'env{i}_cup_inspection')
            action=self.inspections[i].step(o)
            if action is None:ready.append(o)
            else:self.inspection_actions[i]=action
        if ready:super().update_obs_batch(ready)
    def get_action_batch(self,ids=None):
        requested=self.latest_ids if ids is None else list(ids)
        if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(self.latest_ids):
            raise ValueError('Requested environments must belong to the current observation batch')
        ready=[i for i in requested if i not in self.inspection_actions]
        actions=dict(zip(ready,super().get_action_batch(ready))) if ready else {}
        result=[self.inspection_actions[i] if i in self.inspection_actions else actions[i] for i in requested]
        self.latest_ids=[]
        return result


Model = ObservedCupModel
