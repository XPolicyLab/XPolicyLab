"""Offline validation only. Does not start an environment, model service or API request."""
import argparse,json,os,hashlib
from pathlib import Path
from .runtime import ROOT,configure,config
from .jev.network import EpisodeRouter

def main():
    p=argparse.ArgumentParser();p.add_argument("--task",required=True);p.add_argument("--assets-only",action="store_true");a=p.parse_args()
    configure();cfg=config({"task_name":a.task})
    from .icl.loader import load_messages
    load_messages(a.task)
    if not a.assets_only:
        checkpoint=os.environ.get("WRSI_JEV_CHECKPOINT") or str(ROOT/cfg["jev"]["checkpoint"])
        route=EpisodeRouter(os.environ.get("WRSI_ROUTE_MODE",cfg["jev"]["mode"]),checkpoint)
        if route.mode in ("learned","astra"):
            from .harness.vlm_client import load_vlm_profile
            load_vlm_profile(os.environ.get("VLM_AGENT_PROFILE") or cfg["vlm_agent"].get("vlm_profile"))
        if route.mode in ("learned","wam"):
            from .wam_backend import build_config
            wam_cfg = build_config(cfg)
            if not wam_cfg.get("checkpoint_path") and not wam_cfg.get("ckpt_name"):
                raise ValueError("Set WRSI_WAM_CHECKPOINT or ckpt_name for WAM")
    print(json.dumps({"offline_preflight":"passed","task":a.task,"assets_only":a.assets_only,"live_simulation_verified":False}))
if __name__=="__main__":main()
