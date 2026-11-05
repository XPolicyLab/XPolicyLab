"""B-15 bounded scheduling retries; B-14 perception/motions unchanged.

Independent implementation of the reviewed support/reobserve and bounded
recovery ideas. No PhysicalRSI source, learned coefficients, reward or layout
state is imported. All costs are conservative planning estimates, not promises.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix
from .contracts import restrict_observation
from .policy import Waypoint


class BottleExecution:
    def reset(self):
        super().reset()
        self.planning_interval=1  # bottle-only: no 0-9 idle steps between plans
        self.cycle_costs=[]
        self.bottle_tracks=[]
        self.active_track=None
        self.view_recoveries=0
        self.progress_key=None
        self.progress_errors=[]
        self.final_home=False
        self.safe_halt=False
        self.staged_hint=None
        self.current_bottle_objects={}
        self.preflight_rounds=0
        self.stop_reason=None
        self.selection_summary={}
        self.held_bottle=None
        self.b17_stats={}

    def remaining_steps(self):
        return max(0,700-self.step)

    def track_source(self,source):
        # Current RGB only. Memory never substitutes for a current detection.
        matches=[(np.linalg.norm(source.xy-t["xy"]),i) for i,t in enumerate(self.bottle_tracks)
                 if abs(source.grasp_height-t["height"])<.03]
        distance,index=min(matches,default=(float("inf"),None))
        if distance>.055:
            index=len(self.bottle_tracks)
            self.bottle_tracks.append(dict(xy=source.xy.copy(),height=source.grasp_height,
                tries=0,status="observed",last_step=-1000,variant=0))
        else:
            self.bottle_tracks[index]["xy"]=source.xy.copy()
        return index

    def estimate_cycle(self,source,side):
        from .bottle_revision17 import estimate_cycle
        return estimate_cycle(self,source,side)

    def ranked_sources(self,objects):
        options=[]
        from .bottle_revision21 import choose_source_side
        self.source_sides={}
        counts=dict(observed=0,exhausted=0,cooling=0,over_budget=0,eligible=0)
        for name,s in objects.items():
            if not name.startswith("bottle") or s.grasp_height is None or s.grasp_height<=.02:continue
            counts["observed"]+=1
            i=self.track_source(s);t=self.bottle_tracks[i]
            if t["tries"]>=3 or t.get('attachment_exhausted',False):
                counts["exhausted"]+=1;continue
            side=choose_source_side(self,s,t["variant"])
            self.source_sides[id(s)]=side
            cost=self.estimate_cycle(s,side)
            self.cycle_costs.append(dict(step=self.step,track=i,side=side,cost=cost,
                remaining=self.remaining_steps(),reserve=60,
                budget_allowed=cost+60<=self.remaining_steps()))
            self.cycle_costs=self.cycle_costs[-16:]
            if cost+60>self.remaining_steps():
                counts["over_budget"]+=1;continue
            if self.step-t["last_step"]<30:
                counts["cooling"]+=1;continue
            counts["eligible"]+=1
            staged=self.staged_hint is not None and np.linalg.norm(s.xy-self.staged_hint)<.055
            # Finish an observed staged object before initiating another transfer.
            # Otherwise prefer a currently visible untried object over repeating
            # the cheapest failed grasp. Existing bounds and cooldown remain.
            options.append((not staged,t["tries"],cost,i,s))
        self.selection_summary=counts
        return [(i,s) for _,_,_,i,s in sorted(options,key=lambda x:x[:4])]

    def begin_source(self,index):
        self.active_track=index
        self.preflight_rounds=0
        t=self.bottle_tracks[index];t["tries"]+=1;t["last_step"]=self.step;t["status"]="attempting"
        self.source_attempts.append(t["xy"].copy())  # historical log, NOT exclusion list

    def finish_source(self,status):
        if self.active_track is not None:
            t=self.bottle_tracks[self.active_track]
            t["status"]=status;t["last_step"]=self.step
            if status=="failed":t["variant"]+=1
        self.active_track=None

    def finish_schedule(self,reason):
        # Exhausted alternatives are not the same as exhausted episode steps.
        self.stop_reason=reason
        self.reason=reason
        if self.home is None:return False
        self.plan=[];self.return_home();self.final_home=True
        self.budget_return=True
        return bool(self.plan)

    def observe_or_finish(self,reason):
        if reason=="no_candidate_or_budget":
            counts=self.selection_summary
            if counts.get("cooling",0)>0:
                self.plan=[];self.reason="bottle_wait_source_cooldown"
                return False
            if counts.get("observed",0)==0:reason="no_current_rgb_candidate"
            elif counts.get("over_budget",0)>0:reason="estimated_cycle_over_budget"
            else:reason="source_attempts_exhausted"
        if self.remaining_steps()<=60:
            return self.finish_schedule("bottle_final_home_reserve")
        # make_plan already advanced the existing grasp-azimuth alternative.
        # Give all three existing choices a chance BEFORE a view/home action.
        # Only a real selected source or completed view plan resets this bound.
        if reason=="path_unqualified" and self.preflight_rounds<2:
            self.preflight_rounds+=1
            self.plan=[];self.reason="bottle_retry_preflight_alternative"
            return False
        # Change the observation before another retry. Never count eight identical
        # observations as eight actual grasps. Both grippers must be empty/open.
        opened=all(float(np.asarray(self.robot_state.get(f"{s}_ee_joint_state",[0.])).reshape(-1)[0])>.9
                   for s in ("left","right"))
        if reason=='estimated_cycle_over_budget' and opened:
            return self.finish_schedule('bottle_alternatives_exhausted_estimated_cycle_over_budget')
        if self.view_recoveries<2 and self.remaining_steps()>160 and opened and self.home is not None:
            poses=[]
            for side in ("left","right"):
                pose=self.home[side].copy()
                pose[0]+=(-1 if side=="left" else 1)*(.06+.03*self.view_recoveries)
                pose[2]+=.05
                stage=Waypoint(side,pose,1.,"bottle_view_clear",3)
                ok,_=self._qualify([stage])
                if ok:poses.append(stage)
            self.view_recoveries+=1
            if poses:
                self.preflight_rounds=0
                self.plan=poses;self.reobserve_after_plan=True
                self.reason="bottle_change_view_after_"+reason
                return True
            if self.view_recoveries<2:
                # A rejected view waypoint did not move either arm. Try the
                # second already-defined view candidate on the next planning
                # opportunity instead of declaring the episode over.
                self.plan=[];self.reason="bottle_retry_view_preflight"
                return False
        if not opened and self.home is not None:
            self.safe_halt=True;self.finished=True;self.plan=[]
            self.stop_reason="bottle_scheduler_hold_unverified_gripper"
            self.reason=self.stop_reason
            return False
        return self.finish_schedule("bottle_alternatives_exhausted_"+reason)

    def receiver_feasible(self,drop,h,radius):
        # Validate the receiving grasp + full carry before donor commits. Fresh
        # RGB is still mandatory after support placement; no blind left grasp.
        from .bottle_policy import exposed_drop_xy
        from .bottle_grasp import upright_descent_clearance
        destination=exposed_drop_xy(radius);clearance=upright_descent_clearance(h,radius)
        if destination is None or clearance is None:return False
        delta=drop-np.array([-.3,-.45]);yaw=float(np.arctan2(delta[1],delta[0]))
        stages=[Waypoint("left",self.pose("left",drop,z,yaw,0.),g,n)
                for z,g,n in [(clearance["hover_height"],1.,"bottle_approach"),
                             (h,1.,"bottle_insert_fingers"),(h+.08,0.,"bottle_lift")]]
        try:stages+=self._transport(drop,destination,h+.08)
        except ValueError:return False
        return self._qualify(stages)[0]

    def recover_lift(self,waypoint):
        pending=self.pending_bottle
        if pending is None or waypoint.name not in {"bottle_lift","bottle_attachment_probe"}:return False
        side=pending["side"];pose=np.asarray(self.robot_state[f"{side}_ee_pose"],float)
        tcp=pose[:3]+quat_matrix(pose[3:])@np.asarray(self.calibration["tool_offset_in_ee"])
        # Only reverse a near-vertical lift at the known RGB pickup support.
        # A held-object carry/staging stall must NOT trigger arbitrary opening.
        if np.linalg.norm(tcp[:2]-pending["xy"])>.03:return False
        yaw=pending["yaw"];pitch=np.pi/2 if pending.get("mode")=="lying" else 0.
        expected=self.pose(side,pending["xy"],pending["height"],yaw,pitch)
        if Rotation.from_matrix(quat_matrix(pose[3:]).T@quat_matrix(expected[3:])).magnitude()>.18:return False
        h=pending["height"]
        self.plan=[]
        for z,g,name in [(h,0.,"bottle_recovery_lower"),(h,1.,"bottle_recovery_open"),
                         (h+.12,1.,"bottle_recovery_retreat")]:
            self.add(side,pending["xy"],z,g,name,5,angle=yaw,pitch=pitch)
        ok,_=self._qualify(self.plan)
        if not ok:self.plan=[];return False
        self.finish_source("failed");self.pending_bottle=None
        self.index=self.age=self.dwell=0;self.finished=False;self.reobserve_after_plan=True
        self.reason="bottle_vertical_lift_recovery"
        return True

    def act(self,raw):
        obs=restrict_observation(raw)
        self.robot_state=obs["state"]
        from .bottle_pregrasp import before_action
        # B16 legacy helper is opt-in only; B17 never initializes pregrasp state.
        if (self.pending_bottle is not None and "pregrasp" in self.pending_bottle
                and before_action(self,obs["vision"]["cam_head"]["color"])):
            # Bounded observation wait or safe hold, never close on stale RGB.
            self.step+=1
            return {f"{side}_{field}":np.asarray(self.robot_state[f"{side}_{field}"],float).copy()
                    for side in ("left","right") for field in ("ee_pose","ee_joint_state")}
        # This override is bottle-only. Parent's attempt count mixes perception
        # and pickup planning; do not let it terminate this finite scheduler.
        if not self.safe_halt and not self.budget_return:
            self.attempts=0
            if not self.plan:self.finished=False
        if not self.safe_halt and not self.plan and self.pending_bottle is None and self.remaining_steps()<=60 and not self.final_home:
            self.return_home();self.final_home=True;self.budget_return=True;self.finished=False
            self.stop_reason="bottle_final_home_reserve"
            self.reason=self.stop_reason
        from .bottle_revision17 import qualify_descent_close,recover_held
        qualify_descent_close(self)
        previous=self.plan[self.index] if self.index<len(self.plan) else None
        from .bottle_revision20 import guard_axial_capture
        guard_axial_capture(self)
        remaining_plan=self.plan[self.index:]
        if previous is not None:
            key=(id(self.plan),self.index)
            pose=np.asarray(self.robot_state[f"{previous.side}_ee_pose"],float)
            error=np.linalg.norm(pose[:3]-previous.pose[:3])
            angle=Rotation.from_matrix(quat_matrix(pose[3:]).T@quat_matrix(previous.pose[3:])).magnitude()
            if key!=self.progress_key:self.progress_key=key;self.progress_errors=[]
            self.progress_errors.append((float(error),float(angle)))
            self.progress_errors=self.progress_errors[-30:]
            # Progress detector, NOT relaxed arrival tolerance. Allow settling.
            if (self.age>=40 and len(self.progress_errors)==30 and
                previous.name not in {"home","bottle_close","bottle_release","bottle_stage_open"} and
                (error>=.009 or angle>=.15)):
                old=self.progress_errors[0]
                if old[0]-error<.002 and old[1]-angle<.02:self.age=160
        action=super().act(raw)
        if self.reason=="motion_timeout" and previous is not None:
            recovered=(self._recover_open_timeout(previous) or self.recover_lift(previous)
                       or recover_held(self,remaining_plan))
            if recovered:
                side=previous.side
                action[f"{side}_ee_pose"]=np.asarray(self.robot_state[f"{side}_ee_pose"],float).copy()
                action[f"{side}_ee_joint_state"]=np.asarray(self.robot_state[f"{side}_ee_joint_state"],float).copy()
            else:
                self.safe_halt=True;self.reason="bottle_stalled_hold_requires_safe_recovery"
                side=previous.side
                action[f"{side}_ee_pose"]=np.asarray(self.robot_state[f"{side}_ee_pose"],float).copy()
                action[f"{side}_ee_joint_state"]=np.asarray(self.robot_state[f"{side}_ee_joint_state"],float).copy()
        self.program_notes["b17"]=dict(self.b17_stats)
        self.program_notes['b23']=dict(planning_interval=self.planning_interval,
            recent_cycle_costs=self.cycle_costs,untried_before_retry=True,
            carry_endpoint_tolerance_unchanged=True)
        self.program_notes["scheduler"]=dict(remaining=self.remaining_steps(),
            tracks=[dict(xy=t["xy"].tolist(),status=t["status"],tries=t["tries"]) for t in self.bottle_tracks],
            views=self.view_recoveries,safe_halt=self.safe_halt,
            preflight_rounds=self.preflight_rounds,selection=self.selection_summary,
            stop_reason=self.stop_reason)
        return action
