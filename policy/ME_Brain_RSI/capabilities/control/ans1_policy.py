"""ANS1: standard RGB digits/circles, retaining ANR6C motion and scheduling."""
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import ARMS,validate_observation,validate_action
from .geometry import tcp_to_ee,bounded_pose,quat_matrix
from .number_perception import NumberPerception,rotation
from .standard_perception import StandardNumberPerception
from .kinematics import SerialArm,pose_matrix
from .execution_order import order_transfers


class Policy:
    def __init__(self,task,calibration):
        if task!="arrange_largest_number": raise ValueError("Wrong candidate task")
        self.task=task; self.calibration=calibration; self.perception=StandardNumberPerception(calibration)
        self.home=None; self.plan=[]; self.index=0; self.age=0; self.dwell=0; self.step=0
        self.reason="initializing"; self.digits=[]; self.targets=[]; self.motion={}; self.planned=False
        self.arms={s:SerialArm.public_x5(s) for s in ARMS}; self.ik_cache={}; self.joints={}; self.transfers=[]
        self.execution_schedule=[]

    def pose(self,xy,z,yaw):
        r=Rotation.from_euler("z",yaw).as_matrix()@Rotation.from_euler("y",np.pi/2).as_matrix()
        return tcp_to_ee([*xy,self.calibration["table_z"]+z],r,self.calibration["tool_offset_in_ee"])

    def add(self,side,xy,z,yaw,gripper,name,dwell=2):
        self.plan.append(dict(side=side,pose=self.pose(xy,z,yaw),gripper=gripper,name=name,dwell=dwell))

    def feasible(self,side,xy,z,yaw):
        key=(side,*np.round(xy,4),round(z,4),round(yaw,4))
        if key not in self.ik_cache:
            # Static feasibility search is independent of the current IK branch;
            # observed joints may seed a local minimum in this coarse prior.
            fit=self.arms[side].inverse(pose_matrix(self.pose(xy,z,yaw)),restarts=3)
            self.ik_cache[key]=fit["position_error_m"]<.004 and fit["orientation_error_rad"]<.08
        return self.ik_cache[key]

    def options(self,xy,goal,yaw):
        choices=[]
        for side in ARMS:
            # Numerical IK does not certify a collision-free cross-arm path.
            # Keep each arm on its own half except a small shared centre strip.
            if side=="left" and max(xy[0],goal[0])>.06: continue
            if side=="right" and min(xy[0],goal[0])<-.06: continue
            # Prefer 100mm transit clearance: 150mm can push the natural
            # elbow branch near its reach limit above the centre pads.
            for height in (.10,.15):
                if all(self.feasible(side,p,z,a) for p,z,a in [(xy,.019,yaw+np.pi/2),(xy,height,yaw+np.pi/2),
                                                             (goal,.024,np.pi/2),(goal,height,np.pi/2)]):
                    base=np.array([-.3 if side=="left" else .3,-.45])
                    choices.append((float(np.linalg.norm(xy-base)+np.linalg.norm(goal-base))+(0 if height==.10 else .03),side,height))
                    break
        return sorted(choices)

    def transfer(self,side,xy,goal,yaw,height,label):
        # Consecutive transfers by the same arm can leave from the raised
        # retreat directly. Always park before changing the active arm.
        if self.plan and self.plan[-1]["name"].startswith("home_") and self.plan[-1]["side"]==side:
            self.plan.pop()
        angle=yaw+np.pi/2
        self.add(side,xy,height,angle,1.,"approach_"+label)
        self.add(side,xy,.019,angle,1.,"descend_"+label,3)
        self.add(side,xy,.019,angle,0.,"close_"+label,10)
        self.add(side,xy,height,angle,0.,"lift_"+label)
        self.add(side,goal,height,np.pi/2,0.,"carry_"+label)
        self.add(side,goal,.024,np.pi/2,0.,"place_"+label,4)
        self.add(side,goal,.024,np.pi/2,1.,"release_"+label,7)
        self.add(side,goal,height,np.pi/2,1.,"retreat_"+label)
        self.plan.append(dict(side=side,pose=self.home[side].copy(),gripper=1.,name="home_"+label,dwell=3))
        self.transfers.append(dict(side=side,digit=label,source=np.asarray(xy).tolist(),target=np.asarray(goal).tolist(),height=height))

    def staging(self,rgb):
        image=self.perception.rectify(rgb,0.)
        for xy in ([0.,-.24],[0.,-.20],[0.,-.16]):
            x,y=np.rint((np.array(xy)-np.array(self.perception.bounds)[[0,2]])*self.perception.scale).astype(int)
            patch=image[y-35:y+36,x-35:x+36].astype(float)
            # Candidate must visibly resemble an unoccupied, low-texture tabletop.
            if patch.shape[:2]==(71,71) and patch.mean()>35 and np.std(patch,axis=(0,1)).max()<18:
                return np.array(xy)
        return None

    def build(self,rgb):
        self.digits,self.targets=self.perception.detect(rgb)
        if len(self.digits) not in (4,5) or len(self.targets)!=len(self.digits):
            self.reason="perception_abstained"; return
        routes=[]
        # Assignment is ALWAYS descending digit -> left-to-right target. Only
        # execution order changes; build the same routes/IK choices as ANR4 in
        # the same initial order before evaluating their scheduling risk.
        for target_index,(digit,target) in enumerate(zip(sorted(self.digits,key=lambda d:-d["digit"]),self.targets)):
            xy=np.array(digit["xy"]); goal=np.array(target["xy"]); yaw=digit["yaw"]
            label=str(digit["digit"])
            choices=self.options(xy,goal,yaw)
            if choices:
                _,side,height=choices[0]
                legs=[dict(side=side,xy=xy,goal=goal,yaw=yaw,height=height,label=label)]
            else:
                middle=self.staging(rgb)
                first=[] if middle is None else self.options(xy,middle,yaw)
                second=[] if middle is None else self.options(middle,goal,0.)
                if not first or not second:
                    self.reason="no_reachable_rgb_clear_transfer"; continue
                _,side,height=first[0]
                legs=[dict(side=side,xy=xy,goal=middle,yaw=yaw,height=height,label=label+"_stage")]
                _,side,height=second[0]
                legs.append(dict(side=side,xy=middle,goal=goal,yaw=0.,height=height,label=label))
            routes.append(dict(digit=digit,target_index=target_index,legs=legs))
        scheduled=order_transfers(routes,self.perception.digits)
        self.execution_schedule=[dict(digit=r["digit"]["digit"],target_index=r["target_index"],**r["schedule_features"]) for r in scheduled]
        for route in scheduled:
            for leg in route["legs"]:
                self.transfer(**leg)
        self.reason="executing_rgb_plan" if self.plan else "no_reachable_rgb_clear_transfer"

    def act(self,obs):
        validate_observation(obs); self.step+=1; state=obs["state"]
        if self.home is None:
            self.home={s:np.array(state[f"{s}_ee_pose"],float) for s in ARMS}
            self.joints={s:np.array(state[f"{s}_arm_joint_state"],float) for s in ARMS if f"{s}_arm_joint_state" in state}
        action={}
        for side in ARMS:
            action[f"{side}_ee_pose"]=np.array(state[f"{side}_ee_pose"],float)
            action[f"{side}_ee_joint_state"]=np.array(state.get(f"{side}_ee_joint_state",[1.]),float)
        if not self.planned:
            self.build(obs["vision"]["cam_head"]["color"]); self.planned=True
        if self.index<len(self.plan):
            w=self.plan[self.index]; key=f'{w["side"]}_ee_pose'; current=action[key]
            precision=w["name"].startswith(("descend","close","place","release"))
            translation=self.calibration["precision_translation_step"] if precision else self.calibration["max_translation_step"]
            action[key]=bounded_pose(current,w["pose"],translation,self.calibration["max_rotation_step"])
            action[f'{w["side"]}_ee_joint_state']=np.array([w["gripper"]])
            pos=float(np.linalg.norm(current[:3]-w["pose"][:3]))
            rot=float(Rotation.from_matrix(quat_matrix(current[3:]).T@quat_matrix(w["pose"][3:])).magnitude())
            self.motion=dict(position_error=pos,orientation_error=rot,side=w["side"],current_ee=current.tolist(),goal_ee=w["pose"].tolist())
            self.age+=1; self.dwell=self.dwell+1 if pos<(.005 if precision else .009) and rot<.12 else 0
            if self.dwell>=w["dwell"]: self.index+=1; self.age=self.dwell=0
            elif self.age>100:
                # Do not proceed to a blind release after stalled robot motion.
                self.reason="motion_timeout_hold"; self.index=len(self.plan)
        elif self.plan and self.reason!="motion_timeout_hold": self.reason="program_complete_not_success_claim"
        validate_action(action); return action

    def diagnostics(self):
        return dict(candidate="ANS1",step=self.step,reason=self.reason,waypoint=self.index,
            phase=self.plan[self.index]["name"] if self.index<len(self.plan) else None,
            digits=self.digits,targets=self.targets,motion=self.motion,transfers=self.transfers,
            execution_schedule=self.execution_schedule,perception=self.perception.last_diagnostics)
