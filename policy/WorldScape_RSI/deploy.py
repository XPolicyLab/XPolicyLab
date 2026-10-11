"""XPolicyLab simulation-side entry; retains our existing primitives and motion executor."""
import os,uuid,json
from pathlib import Path
from .runtime import configure,config

def eval_one_episode(TASK_ENV,model_client):
    configure()
    cfg=config(getattr(TASK_ENV,"deploy_cfg",{}))
    task=getattr(TASK_ENV,"task_name",None) or cfg["task_name"]
    if task!=cfg["task_name"]:raise ValueError("Environment and policy task mismatch")
    from .runtime import release_identity
    metadata=model_client.call(func_name="release_metadata")
    if metadata.get("release_sha256")!=release_identity() or metadata.get("task")!=task:
        raise RuntimeError("Server/client release or task mismatch")
    # Explicit episode boundary, before route selection and execution reports.
    model_client.call(func_name="reset")
    run=Path(os.environ.get("WRSI_OUTPUT_DIR","eval_result/WorldScape_RSI"))/task/uuid.uuid4().hex
    run.mkdir(parents=True,exist_ok=False)
    (run/"release.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    os.environ["RDA_TASK"]=task;os.environ["RDA_RUN_DIR"]=str(run.resolve())
    from .harness import deploy
    # Keep server/client execution settings identical; no launch-directory dependent YAML.
    TASK_ENV.deploy_cfg=cfg
    deploy.eval_one_episode(TASK_ENV,model_client)

def eval_batch(*args,**kwargs):raise NotImplementedError("One process per environment")
