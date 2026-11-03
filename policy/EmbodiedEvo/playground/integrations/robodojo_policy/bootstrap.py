"""Load an external policy namespace without modifying the vendor snapshot."""

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import sys
import traceback
import types
from unittest.mock import patch

from orchestrator.config import CODE_ROOT, REPO

ROOT = CODE_ROOT
VENDOR = REPO / "vendor/robodojo"


def imports():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(VENDOR), str(VENDOR / "XPolicyLab"), str(ROOT)]
    import XPolicyLab
    try:
        import XPolicyLab.policy
    except ModuleNotFoundError as exc:
        if exc.name != "XPolicyLab.policy":
            raise
        # The provisioned eval-only snapshot omits the policy zoo.
        package = types.ModuleType("XPolicyLab.policy")
        package.__path__ = []
        sys.modules[package.__name__] = package
        XPolicyLab.policy = package
    path = str(Path(__file__).parent / "policy")
    if path not in XPolicyLab.policy.__path__:
        # This subprocess serves the native Playground transport. Prefer it to
        # the outer Test adapter, which intentionally has the same policy name.
        XPolicyLab.policy.__path__.insert(0, path)


async def serve(spec):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.policy.EmbodiedEvo.model import Model
    from orchestrator.files import write_json
    model = Model(spec)
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=spec["port"]))
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)
    try:
        await server.start()
        write_json(Path(spec["run_dir"]) / "policy-server.json", {
            "status": "ready", "pid": os.getpid(), "url": server.url,
        })
        await stopped.wait()
    finally:
        model.close()
        await server.stop()


def create_native_env(factory, config, app, **kwargs):
    if os.environ.get("EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER") == "1":
        from simulators.robodojo_startup import wait_for_renderer

        wait_for_renderer(app)
    return factory(config, app, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "client"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--transport-socket")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--video-threads", type=int, default=0)
    parser.add_argument("--collision-stack-bytes", type=int, default=0)
    parser.add_argument("--rigid-patch-count", type=int, default=0)
    parser.add_argument("--reuse-reset-rigid-bodies", action="store_true")
    parser.add_argument("--disable-retired-collisions", action="store_true")
    parser.add_argument("--reserve-monitor-pipe", action="store_true")
    parser.add_argument("--gpu")
    args = parser.parse_args(argv)
    spec = json.loads((args.run_dir / "adapter.json").read_text())
    imports()
    if args.transport_socket:
        from integrations.robodojo_policy.transport import forward
        remaining = list(sys.argv[1:] if argv is None else argv)
        index = remaining.index("--transport-socket")
        del remaining[index:index + 2]
        with forward(("127.0.0.1", spec["port"]),
                     (socket.AF_UNIX, args.transport_socket)):
            return main(remaining)
    from orchestrator.files import write_json
    try:
        if args.mode == "server":
            asyncio.run(serve(spec))
            return
        if (args.reuse_reset_rigid_bodies or args.disable_retired_collisions) and spec.get("phase") != "playground":
            raise ValueError("Fixed-scene rigid reuse is only supported in Playground")
        if args.cpu_threads:
            os.environ["EMBODIEDEVO_ROBODOJO_KIT_ARGS"] = (
                os.environ.get("EMBODIEDEVO_ROBODOJO_KIT_ARGS", "")
                + f" --/plugins/carb.tasking.plugin/threadCount={args.cpu_threads}"
            )
        if args.video_threads:
            os.environ["EMBODIEDEVO_ROBODOJO_VIDEO_THREADS"] = str(args.video_threads)
        os.chdir(args.run_dir / "official")
        os.environ["EVAL_NUM"] = str(spec["episodes"])
        os.environ["ROBODOJO_RUN_ID"] = args.run_dir.name
        # This smoke run reports infrastructure errors rather than silently
        # replacing scenes through self-restarting command lines.
        os.environ["ROBODOJO_FATAL_RESTART_COUNT"] = "3"
        Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
        sys.argv = [str(VENDOR / "src/eval_client/main.py"),
                    "--task_name", spec["env_task"], "--num_envs", "1",
                    "--env_cfg_type", "arx_x5", "--device_id", "0",
                    "--policy_name", "EmbodiedEvo", "--host", "127.0.0.1",
                    "--port", str(spec["port"]), "--protocol", "ws",
                    "--seed", str(spec["seed"]),
                    "--additional_info", "ckpt_name=frozen_harness,action_type=ee",
                    "--headless", "--enable_cameras"]
        if os.environ.get("EMBODIEDEVO_ROBODOJO_KIT_ARGS"):
            sys.argv += ["--kit_args", os.environ["EMBODIEDEVO_ROBODOJO_KIT_ARGS"]]
        if args.ready_fd is not None:
            os.write(args.ready_fd, b'{"event":"client_starting"}\n')
        write_json(args.run_dir / "official-client.json", {"status": "running", "pid": os.getpid()})
        # Initialize the upstream module, then call its unmodified main loop.
        # The local IsaacLab build requires an explicit logger directory.
        # This hook changes logging only, and makes setup errors fail loudly
        # instead of the upstream loop silently consuming another layout.
        if args.reserve_monitor_pipe:
            from src.eval_client.physx_warning_monitor import PhysXWarningMonitor
            from integrations.robodojo_policy.pipe_capacity import reserve_monitor_pipe

            PhysXWarningMonitor.DESIRED_PIPE_BYTES = 64 * 1024
            PhysXWarningMonitor._enlarge_pipe = lambda monitor, fd: reserve_monitor_pipe(
                fd, args.run_dir / "pipe-reservations.jsonl")
        namespace = runpy.run_path(str(VENDOR / "src/eval_client/main.py"),
                                  run_name="embodiedevo_official_client")
        native_main = namespace["main"]
        native_factory = namespace["create_eval_env"]
        if args.video_threads:
            from simulators.robodojo_video import configure_video_threads
            import utils.save_file as video_module
            configure_video_threads(video_module, args.video_threads)
        created = []

        class PlaygroundFinished(BaseException):
            pass

        def create_env(*positional, **keywords):
            from XPolicyLab.policy.EmbodiedEvo.deploy import AdapterFailure
            from simulators.robodojo_capture import install_capture_cleanup

            env = create_native_env(native_factory, *positional, **keywords)
            created.append(env)
            install_capture_cleanup(env.capture_manager)
            setup = env.setup_sim_cfg
            reset = env.reset
            reset_count = 0
            if "layout_ordinal" in spec:
                # A run-local subset chosen by the host after freezing H.
                # Each process still initializes exactly one native scene.
                pending_layouts = [[spec["layout_ordinal"]]]
                env.seed_manager.get_seeds = lambda **_: (
                    pending_layouts.pop() if pending_layouts else None
                )
            if spec.get("phase") == "playground" or spec.get("diagnostic"):
                from XPolicyLab.policy.EmbodiedEvo.deploy import playground_one_session

                def run_playground():
                    playground_one_session(env, env.model_client)
                    raise PlaygroundFinished()

                env.run_eval = run_playground

            def setup_with_log_directory(config):
                from simulators.robodojo_startup import configure_collision_stack, configure_rigid_patch_count

                setup(config)
                env.sim_cfg.sim.log_dir = str(args.run_dir / "official/isaaclab-logs")
                capacity = configure_collision_stack(env.sim_cfg.sim.physx, args.collision_stack_bytes)
                capacity["rigid_patch_count"] = configure_rigid_patch_count(
                    env.sim_cfg.sim.physx, args.rigid_patch_count)
                write_json(args.run_dir / "native-capacity.json", capacity)
                write_json(args.run_dir / "native-reset-config.json", {
                    "reuse_reset_rigid_bodies": args.reuse_reset_rigid_bodies,
                    "disable_retired_collisions": args.disable_retired_collisions,
                    "phase": spec["phase"],
                })

            def checked_reset(*values, **options):
                nonlocal reset_count
                try:
                    # Reuse EmbodiedEvo's native reset repairs; no second simulator
                    # or alternate controller is constructed here.
                    from simulators.robodojo import RoboDojoTask
                    repair = types.SimpleNamespace(
                        native=env, _restoring_scene=reset_count > 0,
                        reuse_reset_rigid_bodies=args.reuse_reset_rigid_bodies,
                        disable_retired_collisions=args.disable_retired_collisions,
                    )
                    repair._check_reset_robot_state = lambda: RoboDojoTask._check_reset_robot_state(repair)
                    client = env.model_client
                    resetting_playground = reset_count > 0 and spec.get("phase") == "playground"
                    if resetting_playground and env.physx_monitor_enabled:
                        # Internal Playground resets bypass main.py's round
                        # boundary. Drain old warnings; retain fatal errors.
                        env._physx_get_monitor().reset()

                    def preserve_worker(*args, **kwargs):
                        name = kwargs.get("func_name", args[0] if args else None)
                        return None if name == "reset" else client.call(*args, **kwargs)

                    policy_guard = (patch.object(env, "model_client", types.SimpleNamespace(call=preserve_worker))
                                    if resetting_playground else contextlib.nullcontext())
                    with (RoboDojoTask._preserve_reset_articulation_names(repair),
                          RoboDojoTask._preserve_reset_articulation_defaults(repair),
                          RoboDojoTask._retire_reset_rigid_bodies(repair),
                          RoboDojoTask._restore_robot_state_on_reset(repair), policy_guard):
                        result = reset(*values, **options)
                    reset_count += 1
                    if env.end_flag[0]:
                        raise RuntimeError("Native layout did not initialize successfully")
                    if spec.get("diagnostic"):
                        from orchestrator.files import append_jsonl
                        from simulators.robodojo_capture import capture_resource_counts

                        append_jsonl(args.run_dir / "capture-resources.jsonl", {
                            "reset_count": reset_count, **capture_resource_counts(env.capture_manager),
                        })
                    return result
                except Exception as exc:
                    write_json(args.run_dir / "native-error.json", {"traceback": traceback.format_exc()})
                    raise AdapterFailure("Official scene initialization failed") from exc

            env.setup_sim_cfg = setup_with_log_directory
            env.reset = checked_reset
            write_json(args.run_dir / "official-client.json", {
                "status": "native_created", "pid": os.getpid(),
                "native_step_limit": int(env.step_lim),
            })
            return env

        native_main.__globals__["create_eval_env"] = create_env
        try:
            native_main()
        except PlaygroundFinished:
            write_json(args.run_dir / "playground-client-result.json", {"finished": True})
            for env in created:
                with contextlib.suppress(Exception):
                    env.model_client.close()
                    env.close()
            namespace["simulation_app"].close()
        write_json(args.run_dir / "official-client.json", {"status": "finished"})
    except BaseException:
        write_json(args.run_dir / f"{args.mode}-error.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
