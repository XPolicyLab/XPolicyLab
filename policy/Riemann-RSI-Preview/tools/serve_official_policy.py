"""One official WebSocket server for the registered Rule and learned tasks."""
import argparse
import asyncio
import json
from pathlib import Path
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_imports():
    from rsi.policy_catalog import runtime_settings
    from rsi.resource_locations import relocate_path
    sdk = Path(runtime_settings()['official_root'])
    sys.path[:0] = [str(ROOT), str(sdk), str(sdk / 'XPolicyLab'), str(sdk / 'third_party/curobo'),
                   str(Path(relocate_path('resource://protocol_dependencies/site')))]
    from rsi.adapters import register_policy
    register_policy()
    return sdk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', '--task_name', dest='mode', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--action_type', choices=('joint', 'ee'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--bench_name', default='RoboDojo')
    parser.add_argument('--env_cfg_type', default='arx_x5')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--physical-gpus', help='Physical GPU indices supplied by the split entry')
    args = parser.parse_args(argv)
    if args.physical_gpus is not None:
        from rsi.gpu_resources import configure_split_devices
        configure_split_devices(args.physical_gpus)
    from rsi.policy_catalog import resolve_task
    mode, route = resolve_task(None, args.mode)
    if route['backend'] == 'rule':
        from rsi.rule_configuration import require_python, settings
        import os
        target = settings()['python']
        if Path(sys.prefix).resolve() != Path(target).parent.parent.resolve():
            os.execv(target, [target, '-B', str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        require_python()
    configure_imports()
    from tools.official_rule_integrity import verify_policy
    verify_policy()  # External SDK/protocol identities only; source is Git-managed.
    output = args.output or Path(tempfile.mkdtemp(prefix='riemann-policy-')) / 'policy'
    if args.config and args.config.suffix in ('.yaml', '.yml'):
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
    else:
        cfg = json.loads(args.config.read_text()) if args.config else {}
    cfg.update(task_name=mode, env_cfg_type=args.env_cfg_type, bench_name=args.bench_name,
               seed=args.seed, protocol='ws', output_dir=str(output),
               action_type=args.action_type or ('ee' if route['backend'] == 'learned' else 'joint'))
    if args.checkpoint:
        cfg['ckpt_name'] = args.checkpoint
    from rsi.official_adapter import Model
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    model = Model(cfg)

    async def serve():
        current = asyncio.current_task()
        loop = asyncio.get_running_loop()
        shutdown_requested = False

        def request_shutdown():
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                current.cancel()

        loop.add_signal_handler(signal.SIGTERM, request_shutdown)
        server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port,
            ws_ping_interval_s=cfg.get('ws_ping_interval_s', 20.0),
            ws_ping_timeout_s=cfg.get('ws_ping_timeout_s', 20.0)))
        try:
            await server.start()
            print(json.dumps({'event': 'official_rule_policy_ready' if route['backend'] == 'rule' else 'official_policy_ready', 'mode': mode,
                              'backend': route['backend'], 'url': server.url}), flush=True)
            await server.serve_forever()
        except asyncio.CancelledError:
            if not shutdown_requested:
                raise
        finally:
            loop.remove_signal_handler(signal.SIGTERM)
            try:
                model.close()
            finally:
                await server.stop()
    asyncio.run(serve())


if __name__ == '__main__':
    main()
