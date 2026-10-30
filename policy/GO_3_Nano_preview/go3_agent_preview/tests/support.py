import numpy as np
DIMS={"arm_dim":[6,6],"ee_dim":[1,1]}
def cfg(task="cover_blocks"):
    return {"task_name":task,"env_cfg_type":"arx_x5","action_type":"joint","image_size":[32,48],"action_horizon":3}
def observation(env_idx=0,instruction="Synthetic test instruction"):
    state={}
    for arm in ("left","right"):
        state[f"{arm}_arm_joint_state"]=np.zeros(6,dtype=np.float32)
        state[f"{arm}_ee_joint_state"]=np.ones(1,dtype=np.float32)
        state[f"{arm}_ee_pose"]=np.array([0,0,1,1,0,0,0],dtype=np.float32)
    return {"env_idx":env_idx,"instruction":instruction,"state":state,"vision":{name:{"color":np.full((60,90,3),240,dtype=np.uint8)} for name in ("cam_head","cam_left_wrist","cam_right_wrist")}}
class Backend:
    def __init__(self,fail=False):self.calls=0;self.fail=fail;self.observed=[]
    def infer(self,**kwargs):
        self.calls+=1;self.observed.append(kwargs)
        if self.fail:raise TimeoutError("synthetic backend timeout")
        return {"result":{"left_arm":[[.1*n]*6 for n in (1,2,3)],"right_arm":[[0]*6]*3,"left_effector":[.4]*3,"right_effector":[.8]*3}}
