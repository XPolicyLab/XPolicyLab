"""XPolicyLab server entry. Lazy Astra creation avoids API requirements on WAM-only runs."""
import os,threading,types
from XPolicyLab.model_template import ModelTemplate
from .runtime import ROOT,configure,config
from .jev.network import EpisodeRouter

class Model(ModelTemplate):
    def __init__(self,model_cfg):
        configure();self.model_cfg=config(model_cfg);os.environ["RDA_TASK"]=self.model_cfg["task_name"]
        j=self.model_cfg["jev"];path=os.environ.get("WRSI_JEV_CHECKPOINT") or j.get("checkpoint")
        if path and not os.path.isabs(path):path=str(ROOT/path)
        self.router=EpisodeRouter(os.environ.get("WRSI_ROUTE_MODE",j["mode"]),path)
        self._astra=None;self._wam=None;self._reports=[];self._obs=None;self._lock=threading.RLock()
    def _ensure_wam(self):
        if not self.router.decision or self.router.decision["executor"] != "wam":
            raise RuntimeError("WAM call outside selected episode route")
        if self._wam is None:
            from .wam_backend import create
            self._wam = create(self.model_cfg)
        return self._wam
    def wam_update_obs(self, obs):
        with self._lock:return self._ensure_wam().update_obs(obs)
    def wam_get_action(self):
        with self._lock:return self._ensure_wam().get_action()
    def wam_commit_action(self, obs):
        with self._lock:return self._ensure_wam().commit_action(obs)
    def wam_discard_action(self, obs):
        with self._lock:
            if self._wam is not None:return self._ensure_wam().discard_action(obs)
    def close(self):
        with self._lock:
            if self._wam is not None:self._wam.close()
    def _ensure_astra(self):
        if self._astra is None:
            from .harness.model import Model as Astra
            from .icl.loader import load_messages
            self._astra=Astra(self.model_cfg)
            def ensure(agent):
                if agent.icl_msgs is None:
                    agent.icl_msgs=load_messages(agent.task_name);agent.icl_demos=[]
            self._astra.agent._ensure_icl=types.MethodType(ensure,self._astra.agent)
            self._astra.reset()
            for report in self._reports:self._astra.report_execution(report)
            self._reports=[]
        return self._astra
    def select_executor(self,obs):
        with self._lock:
            d=self.router.select(obs)
            if d["executor"]=="astra":
                from .harness import skill_pool
                d.update(skills=["verify_recover"],skill_pool_sha256=skill_pool.load()[1])
                self._ensure_astra().agent.execution_skill_text=skill_pool.guidance(d)
            else:d["skills"]=["continuous_visual_control"]
            return d
    def reset(self):
        with self._lock:
            self.router.reset();self._reports=[];self._obs=None
            if self._astra is not None:self._astra.reset()
            if self._wam is not None:self._wam.reset()
    def release_metadata(self):
        from .runtime import release_identity
        return {"release_sha256":release_identity(),"task":self.model_cfg["task_name"],
                "route_mode":self.router.mode,"jev_checkpoint_sha256":self.router.digest}
    def update_obs(self,obs):
        with self._lock:
            if not self.router.decision or self.router.decision["executor"]!="astra":raise RuntimeError("Astra observation outside selected route")
            from .harness.observation_contract import filter_observation
            self._obs=filter_observation(obs)
            # Preserve our existing measured-finger feedback (not object truth or reward).
            for key in ("gripper_real","gripper_closed_floor","calibration","policy_budget_s"):
                if key in obs:self._obs[key]=obs[key]
            self._ensure_astra().update_obs(self._obs)
    def get_action(self):
        with self._lock:
            if not self.router.decision or self.router.decision["executor"]!="astra" or self._obs is None:raise RuntimeError("No Astra route/observation")
            return self._ensure_astra().get_action()
    def report_execution(self,obs):
        with self._lock:
            clean={k:v for k,v in obs.items() if k not in ("attack_feedback","process_score","score","reward")}
            if self._astra is not None:return self._astra.report_execution(clean)
            self._reports.append(clean);return {"ok":True}
    def update_obs_batch(self,*args,**kwargs):raise NotImplementedError("One process per environment")
    def get_action_batch(self,*args,**kwargs):raise NotImplementedError("One process per environment")
