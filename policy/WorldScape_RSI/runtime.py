"""Deployment configuration, shared by server and simulation client."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def release_identity():
    import hashlib
    return hashlib.sha256((ROOT/"MANIFEST.json").read_bytes()).hexdigest()

def configure():
    forbidden={"RDA_ADAPTIVE":"1","RDA_ATTACK_MODE":"1","ROBODAWN_PRIVILEGED_REWARD_FEEDBACK":"1",
               "ROBODAWN_REWARD_FEEDBACK":"1","RSI5_PRIVILEGED_REWARD_FEEDBACK":"1","RSI6_REWARD_FEEDBACK":"1"}
    for key,value in forbidden.items():
        if os.environ.get(key)==value:raise RuntimeError("Disable experimental override: "+key)
    for key in ("ROBODAWN_DESKTOP_BRIDGE","RDA_WORKER_ID","VLM_AGENT_OVERRIDES","ROBODAWN_SFT_DIR"):
        if os.environ.get(key):raise RuntimeError("Unattended release rejects override: "+key)
    os.environ.update(RDA_ADAPTIVE="0",RDA_ATTACK_MODE="0",RDA_STAGE="final",RDA_EXECUTOR="manager",
                      ROBODAWN_TASK_SKILLS="1",ROBODAWN_TASK_SKILL_ROOT=str(ROOT/"skills"))

def config(supplied=None):
    import yaml
    base=yaml.safe_load((ROOT/"deploy.yml").read_text(encoding="utf-8"))
    for k,v in (supplied or {}).items():
        if v is not None:
            if isinstance(v,dict) and isinstance(base.get(k),dict):base[k].update(v)
            else:base[k]=v
    if base.get("eval_batch"):raise ValueError("Use one policy process per environment; batch unsupported")
    task=base.get("task_name") or os.environ.get("RDA_TASK")
    if task not in {t["task"] for t in json.loads((ROOT/"configs/tasks.json").read_text())}:raise ValueError("task_name must be a registered task")
    base["task_name"]=task
    base["vlm_agent"]["icl"]={"demo_bank":str(ROOT/"icl"),"demo_count":1}
    return base
