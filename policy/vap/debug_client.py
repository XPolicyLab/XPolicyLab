"""Run the unchanged official TestEnv with this slow policy's configured timeouts."""
import argparse, os, time
from pathlib import Path
import yaml
from XPolicyLab.utils.debug_env_client import TestEnv

def main():
    parser=argparse.ArgumentParser()
    for name in ('bench','task','env_cfg','host','port'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=yaml.safe_load(Path(__file__).with_name('deploy.yml').read_text())
    run_id = os.environ.get('ROBODOJO_RUN_ID',
                            f'vap_{args.task}_{time.strftime("%Y%m%dT%H%M%S")}_{os.getpid()}')
    config.update(bench_name=args.bench,task_name=args.task,env_cfg_type=args.env_cfg,
                  host=args.host,port=int(args.port),policy_server_url=None,
                  evaluation_id=run_id,trial_id=run_id,
                  action_case_id=None,repeat_index=None,
                  obs_encoded=os.environ.get('DEBUG_OBS_ENCODED','0').lower() in ('1','true','yes'))
    env=TestEnv(config)
    try:
        for index in range(10):
            print('[OFFICIAL_DEBUG] episode',index,'encoded',config['obs_encoded'],flush=True)
            env.reset()
            env.eval_one_episode()
            env.finish_episode()
    finally:env.model_client.close()
    print('[OFFICIAL_DEBUG] completed 10 episodes',flush=True)

if __name__=='__main__':main()
