"""Load the Riemann-RSI-Preview policy plugin and execute the unmodified official CLI."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.rule_configuration import settings, require_python, requested_mode
SDK = Path(settings()['official_root'])


def main():
    bootstrap_arguments = [str(Path(__file__).resolve()), *sys.argv[1:]]
    if any(argument in ('--help', '-h') for argument in sys.argv[1:]):
        print('Official policy client: --task_name MODE --policy_name Riemann-RSI-Preview '
              '--env_cfg_type arx_x5 --port PORT --protocol ws [official SDK flags]. '
              'All 54 registered modes share this client. Episode selection and scoring belong to the SDK.')
        return 0
    require_python()
    from rsi.deployment import evaluator_path
    evaluator_path(os.environ, settings()['python'],
                   require_ffmpeg=os.environ.get('EVAL_ENV_TYPE') != 'debug')
    import argparse
    split = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    split.add_argument('--physical-gpus')
    split_args, sys.argv[1:] = split.parse_known_args(sys.argv[1:])
    if split_args.physical_gpus is not None:
        from rsi.gpu_resources import configure_split_devices
        configure_split_devices(split_args.physical_gpus, renderer=True)
    requested = requested_mode(sys.argv[1:])
    from tools.official_rule_integrity import verify_policy
    verify_policy()
    from rsi.resource_locations import relocate_path
    sys.path.insert(0, str(Path(relocate_path('resource://protocol_dependencies/site'))))
    if requested.policy_name is not None and requested.policy_name != settings()['policy_name']:
        raise ValueError('This adapter must keep the Riemann-RSI-Preview policy identity')
    if requested.policy_name is None:
        sys.argv.extend(['--policy_name', settings()['policy_name']])
    # Python's ordinary package search path is extended only for our policy.
    # No evaluator/client/layout/camera/reward class or callback is replaced.
    sys.path[:0] = [str(SDK), str(SDK / 'XPolicyLab')]
    from rsi.adapters import register_policy
    register_policy()
    mode = os.environ.get('EVAL_ENV_TYPE', 'sim')
    if mode not in ('sim', 'debug', ''):
        raise ValueError('Use the official simulation or explicit interface-debug client')
    target = SDK / ('XPolicyLab/utils/debug_env_client.py' if mode == 'debug' else 'src/eval_client/main.py')
    if mode == 'debug':
        import argparse
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        for name in ('bench_name', 'task_name', 'env_cfg_type', 'policy_name', 'protocol', 'host', 'port',
                     'policy_server_url', 'evaluation_id', 'action_case_id', 'trial_id', 'repeat_index',
                     'eval_episode_num', 'eval_batch', 'obs_encoded'):
            parser.add_argument('--' + name)
        args, _ = parser.parse_known_args(sys.argv[1:])
        values = {k: v for k, v in vars(args).items() if v is not None}
        values.setdefault('bench_name', 'RoboDojo')
        sys.argv[1:] = [value for k, v in values.items() for value in ('--' + k, v)]
    else:
        from rsi.isaac_runtime import client_arguments
        sys.argv[1:] = client_arguments(sys.argv[1:])
    sys.argv[0] = str(target)
    original_execute = os.execv
    try:
        if mode != 'debug':
            from rsi.policy_process import route_native_restarts
            # Keep our bootstrap on the SDK's own bounded recovery restart;
            # do not alter its resume state, counters or layouts.
            route_native_restarts(bootstrap_arguments)
        runpy.run_path(str(target), run_name='__main__')
    finally:
        os.execv = original_execute


if __name__ == '__main__': main()
