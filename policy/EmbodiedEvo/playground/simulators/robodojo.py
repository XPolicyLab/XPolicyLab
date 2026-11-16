"""Host-owned Isaac Sim / RoboDojo behind the persistent code interface."""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simulators.execution import CodeTask
from simulators.robodojo_api import RoboDojoLowLevelApi
from simulators.robodojo_layouts import ROOT, decode_scene


class _NoPolicyServer:
    """The socket session supplies actions; no learned-policy server is needed."""

    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        if kwargs.get("func_name") != "reset":
            raise RuntimeError("RoboDojo policy inference is owned by the EmbodiedEvo agent")

    def close(self):
        pass


@contextlib.contextmanager
def robodojo_cwd():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


class RoboDojoTask(CodeTask):
    official_action_limit = True

    def __init__(
        self, task_name, *, seed=None, solve_articulation_contact_last=False,
        min_position_iterations=None, min_velocity_iterations=None,
        official_action_limit=True,
    ):
        super().__init__()
        self.official_action_limit = official_action_limit
        self.solve_articulation_contact_last = solve_articulation_contact_last
        self.min_position_iterations = min_position_iterations
        self.min_velocity_iterations = min_velocity_iterations
        for value in (min_position_iterations, min_velocity_iterations):
            if value is not None and (type(value) is not int or not 1 <= value <= 255):
                raise ValueError("PhysX solver iteration overrides must be integers in [1, 255]")
        self.task_name = task_name
        self.scene_seed = 0 if seed is None else seed
        self.bank, self.ordinal, self.layout_path = decode_scene(task_name, self.scene_seed)
        if ROOT.is_symlink() or not ROOT.is_dir():
            raise RuntimeError("Provision the repository-local vendor/robodojo checkout first")
        sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/robodojo-runtime-{os.getuid()}"))
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Kit holds NFS log files open until shutdown; cleanup must not prevent
        # the application from closing when .nfs files temporarily remain.
        self._temporary = tempfile.TemporaryDirectory(
            prefix="embodiedevo-robodojo-", ignore_cleanup_errors=True
        )
        self.app = self.native = None
        self._physx_monitor = None
        self._failure = None
        self.max_steps = None
        self.initialize(RoboDojoLowLevelApi(self))
        try:
            with robodojo_cwd():
                self._create_native()
            # Keep the task-specific native cutoff in both Playground and Test.
            # The explicit opt-out is only for reproducing historical unlimited runs.
            self.native_step_limit = int(self.native.step_lim)
            if self.official_action_limit:
                self.max_steps = self.native_step_limit
            else:
                self.native.step_lim = float("inf")
        except BaseException:
            self.close()
            raise

    def _create_native(self):
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        from isaaclab.app import AppLauncher
        from utils.load_file import load_yaml

        task_cfg = load_yaml(str(ROOT / "task/RoboDojo/config" / (self.task_name + ".yml")))
        monitor_enabled = bool(task_cfg.get("Articulation"))
        if monitor_enabled:
            from src.eval_client.physx_warning_monitor import get_monitor

            self._physx_monitor = get_monitor()
            self._physx_monitor.start(enabled=True)

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.headless, args.enable_cameras, args.device = True, True, "cuda:0"
        args.kit_args = os.environ.get("EMBODIEDEVO_ROBODOJO_KIT_ARGS", args.kit_args)
        add_zero_delay_kit_args(args)
        self.app = AppLauncher(args).app
        if os.environ.get("EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER") == "1":
            from simulators.robodojo_startup import wait_for_renderer

            self.renderer_startup_sec = wait_for_renderer(self.app)

        import src.eval_client.eval_env as eval_module
        from omegaconf import OmegaConf
        from utils.pipeline_utils import process_config, process_randomization

        if video_threads := int(os.environ.get("EMBODIEDEVO_ROBODOJO_VIDEO_THREADS", "0")):
            import utils.save_file as video_module
            from simulators.robodojo_video import configure_video_threads

            configure_video_threads(video_module, video_threads)

        base = ROOT / "env_cfg"
        eval_cfg = load_yaml(str(base / "arx_x5.yml"))
        eval_cfg.update(
            task_name=self.task_name,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            eval_num=1,
            policy_name="EmbodiedEvo",
            seed=self.bank,
            additional_info="workspace-session-v1",
            physx_monitor_enabled=monitor_enabled,
        )
        cfg = OmegaConf.create(
            {
                "eval_cfg": eval_cfg,
                "deploy_cfg": {"policy_name": "EmbodiedEvo", "port": 1},
                **{
                    name: load_yaml(str(base / name / (eval_cfg["config"][name] + ".yml")))
                    for name in ("sim", "scene", "camera", "robot")
                },
                "task_env": task_cfg,
            }
        )
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [self.ordinal]
        # Native default is CPU state/PhysX with RTX rendering and CUDA CuRobo.
        # Do not change the upstream physics device or rendering quality.
        cfg.sim.device = "cpu"
        cfg, _ = process_config(process_randomization(cfg), self.task_name)
        cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
        with patch.object(eval_module, "WsModelClient", _NoPolicyServer):
            self.native = eval_module.create_eval_env(
                cfg,
                self.app,
                resume_state={"save_dir": self._temporary.name, "details": {}},
            )
        from simulators.robodojo_capture import install_capture_cleanup

        install_capture_cleanup(self.native.capture_manager)
        # Upstream does not forward a YAML log_dir into SimulationCfg. Keep
        # native configuration intact and give its logger a session-owned path.
        configure_scene = self.native.setup_sim_cfg

        def configure_with_private_logs(config):
            configure_scene(config)
            self.native.sim_cfg.sim.log_dir = str(Path(self._temporary.name) / "isaaclab-logs")
            if self.solve_articulation_contact_last:
                # Explicitly recorded in the episode config; task/reward and
                # physical properties are unchanged. Resolve contact after the
                # stiff articulation constraints to avoid numerical divergence.
                self.native.sim_cfg.sim.physx.solve_articulation_contact_last = True
            physx = self.native.sim_cfg.sim.physx
            if self.min_position_iterations is not None:
                physx.min_position_iteration_count = self.min_position_iterations
            if self.min_velocity_iterations is not None:
                physx.min_velocity_iteration_count = self.min_velocity_iterations

        self.native.setup_sim_cfg = configure_with_private_logs

    @contextlib.contextmanager
    def _preserve_reset_articulation_names(self):
        # Native reload keeps articulated objects alive, but regenerates their
        # names in the layout index. Reuse those identities for this fixed scene
        # so stability/reward lookup still addresses the retained physics bodies.
        if not self._restoring_scene:
            yield
            return
        layout = self.native.scene_manager.layout_manager
        kinds = ["Articulation"]
        if getattr(self, "reuse_reset_rigid_bodies", False):
            kinds.append("Rigid")
        names = {}
        for kind in kinds:
            records = layout.object_records_by_type[kind].layout_records_by_env
            for env_idx, objects in enumerate(records):
                for obj in objects:
                    category, model, _ = obj["inst_name"].rsplit("_", 2)
                    names.setdefault((kind.lower(), env_idx, category, int(model)), []).append(
                        (obj["prim_path"], obj["inst_name"])
                    )
        generate = layout._generate_object_paths

        def stable_paths(env_idx, category, model, type):
            if type in {kind.lower() for kind in kinds}:
                return names[(type, env_idx, category, model)].pop(0)
            return generate(env_idx, category, model, type=type)

        with patch.object(layout, "_generate_object_paths", stable_paths):
            yield

    @contextlib.contextmanager
    def _preserve_reset_articulation_defaults(self):
        # Native post_init reinitializes retained objects and captures their
        # manipulated joint positions as defaults. Keep the first completed
        # reset's defaults; native apply_saved_pose still restores the state.
        if not self._restoring_scene:
            yield
            return
        with contextlib.ExitStack() as stack:
            for objects in self.native.scene_manager._articulation_objects:
                for obj in objects.values():
                    defaults = copy.deepcopy(obj.initial_joint_positions)
                    initialize = obj.initialize

                    def preserve(*args, _obj=obj, _defaults=defaults,
                                 _initialize=initialize, **kwargs):
                        result = _initialize(*args, **kwargs)
                        _obj.initial_joint_positions = copy.deepcopy(_defaults)
                        return result

                    stack.enter_context(patch.object(obj, "initialize", preserve))
            yield

    @contextlib.contextmanager
    def _retire_reset_rigid_bodies(self):
        """Move old CPU bodies away before native reload hides their prims."""
        if not self._restoring_scene:
            yield
            return
        manager = self.native.scene_manager
        if getattr(self, "reuse_reset_rigid_bodies", False):
            # Fixed-scene reset can retain rigid actors, as native CUDA reload
            # does. Native relocation and apply_saved_poses still restore state.
            clear, spawn = manager.clear_scene_objects, manager.spawn_scene_objects
            relocate = manager.relocate_stale_objects

            def with_rigid(types):
                return list(dict.fromkeys([*(types or []), "rigid"]))

            def retained_clear(env_id, exclude_types=None):
                return clear(env_id, with_rigid(exclude_types))

            def retained_spawn(env_id, exclude_types=None):
                return spawn(env_id, with_rigid(exclude_types))

            def retained_relocate(env_id, obj_types=None):
                return relocate(env_id, None if obj_types is None else with_rigid(obj_types))

            with (patch.object(manager, "clear_scene_objects", retained_clear),
                  patch.object(manager, "spawn_scene_objects", retained_spawn),
                  patch.object(manager, "relocate_stale_objects", retained_relocate)):
                yield
            return
        delete = manager.delete_scene_object
        rigid = {
            id(obj)
            for objects in manager._rigid_and_dynamic_objects
            for obj in objects.values()
        }
        disable_collisions = getattr(self, "disable_retired_collisions", False)
        retired = set(rigid)
        if disable_collisions:
            retired.update(id(obj) for objects in manager._geometry_objects for obj in objects.values())

        def retire(obj):
            # Native CPU reload hides/disables old rigid bodies without deleting
            # their prims. A later physics reset can revive those stale bodies at
            # their previous poses, overlapping the new saved-layout objects.
            if id(obj) in rigid:
                obj.relocate_offscreen()
            result = delete(obj)
            if disable_collisions and id(obj) in retired:
                from pxr import Usd, UsdPhysics

                prim = obj.stage.GetPrimAtPath(obj.prim_path)
                if prim and prim.IsValid():
                    # Native CPU destroy disables the root view, but collision
                    # shapes can live on descendants. Keep rigid registration
                    # intact so existing physics tensor views remain valid.
                    for child in Usd.PrimRange(prim):
                        if child.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)
            return result

        with patch.object(manager, "delete_scene_object", retire):
            yield

    @contextlib.contextmanager
    def _restore_robot_state_on_reset(self):
        manager = self.native.robot_manager
        set_targets = manager.set_robot_init_pose

        def restore_state():
            # Native reset only sets servo targets. A folded/self-colliding arm
            # may never reach them. Restore both physical state and targets.
            set_targets()
            for robot, articulation in zip(manager.robot_list, manager.robot_key, strict=True):
                if robot.robot_type == "arm":
                    articulation.reset()
                    articulation.write_joint_state_to_sim(
                        articulation.data.default_joint_pos.clone(),
                        articulation.data.default_joint_vel.clone(),
                    )
            # Check the state write before native reset advances physics. The
            # settling steps below can move joints through gravity/contact;
            # equality to the pre-settling pose is not an invariant there.
            self._check_reset_robot_state()

        with patch.object(manager, "set_robot_init_pose", restore_state):
            # Native soft reset/reload can advance physics before robot.reset().
            # Stop the previous motion before those steps; keep the normal
            # native reset hook too. The first reset has no initialized views.
            if self._restoring_scene:
                restore_state()
            yield

    def _check_reset_robot_state(self, *, require_default=True):
        manager = self.native.robot_manager
        for index, (robot, articulation) in enumerate(
            zip(manager.robot_list, manager.robot_key, strict=True)
        ):
            if robot.robot_type != "arm":
                continue
            for name, tolerance in (("joint_pos", 1e-3), ("joint_vel", 1e-2)):
                actual = getattr(articulation.data, name).detach().cpu().numpy()
                if not np.isfinite(actual).all():
                    raise RuntimeError(
                        f"RoboDojo reset produced non-finite robot {index} {name}"
                    )
                if not require_default:
                    continue
                expected = getattr(articulation.data, "default_" + name).detach().cpu().numpy()
                if not np.allclose(actual, expected, atol=tolerance, rtol=0):
                    raise RuntimeError(
                        f"RoboDojo reset failed to restore robot {index} {name}: "
                        f"maximum error {np.max(np.abs(actual - expected))}"
                    )

    def reset(self, *, seed=None, options=None):
        seed = self.scene_seed if seed is None else seed
        if seed != self.scene_seed or options:
            raise ValueError("Each RoboDojo process owns one fixed layout; create a new session")
        try:
            with robodojo_cwd():
                if self._physx_monitor is not None:
                    self._physx_monitor.reset()
                with (
                    self._preserve_reset_articulation_names(),
                    self._preserve_reset_articulation_defaults(),
                    self._restore_robot_state_on_reset(),
                    self._retire_reset_rigid_bodies(),
                ):
                    self.native.reset(seed=[self.ordinal])
                if self.official_action_limit:
                    # Playground reset starts a fresh native action allowance. Keep
                    # cumulative telemetry, while the session's submission counter
                    # and wall-clock budget remain owned by the orchestrator.
                    used = self.control_steps if self._restoring_scene else 0
                    self.native.step_lim = self.native_step_limit
                    self.max_steps = used + self.native_step_limit
                else:
                    self.native.step_lim = float("inf")
                if self.native.episode_nums != 1 or self.native.end_flag[0]:
                    raise RuntimeError("RoboDojo layout failed native stability checks")
                self._check_reset_robot_state(require_default=False)
                self.native.run_reward()
                if hasattr(self.native, "get_score"):
                    self.native.get_score()
                if getattr(self.native, "interact", False) and hasattr(
                    self.native, "query_support_arm_traj"
                ):
                    self.native.query_support_arm_traj(env_idx=0)
                obs = self.native.get_obs()
                if self._restoring_scene:
                    # Native reset regenerates language. Keep the original episode instruction.
                    self.native.obs_manager.instruction[0] = self._task_prompt
                    obs["instruction"] = self._task_prompt
                else:
                    self._task_prompt = obs["instruction"]
            return self.start_episode(seed, obs, reset_options=options)
        except Exception as exc:
            self._failure = exc
            raise

    def control_step(self, action):
        if self.done:
            return self.transition(float(self.success))
        if not isinstance(action, dict) or not action:
            raise ValueError("RoboDojo step requires a native joint or ee action dictionary")
        action = {key: np.asarray(value, dtype=np.float64) for key, value in action.items()}
        if any(value.ndim != 1 or not np.isfinite(value).all() for value in action.values()):
            raise ValueError("RoboDojo actions must contain finite one-dimensional arrays")
        # Native validation has no physics side effects. Reject malformed actions early.
        self.native.validate_action_dict(action)
        mode = self.native.get_action_type(action)
        state = self.public_observation()["state"]
        required = {
            key
            for key in state
            if key.endswith("_ee_joint_state")
            or key == "ee_joint_state"
            or (mode == "joint" and key.endswith("arm_joint_state"))
            or (mode == "ee" and key.endswith("ee_pose"))
        }
        if set(action) != required:
            raise ValueError(f"Expected {mode} action keys {sorted(required)}")
        try:
            with robodojo_cwd():
                previous = int(self.native.take_action_cnt[0])
                self.native.take_action(action)
                self.control_steps += int(self.native.take_action_cnt[0]) - previous
                if 0 in self.native.unstable_envs:
                    raise RuntimeError("Native RoboDojo marked this episode unstable")
                ended = bool(self.native.end_flag[0])
                # success starts True upstream: it means still viable until end_flag.
                self.success = ended and bool(self.native.success[0])
                self.truncated = bool(
                    ended and not self.success and self.official_action_limit
                    and self.control_steps >= self.max_steps
                )
                self.terminated = ended and not self.truncated
                self.observe(self.native.get_obs())
            return self.transition(float(self.success))
        except Exception as exc:
            self._failure = exc
            raise

    def _step(self, code):
        if self._failure is not None:
            raise RuntimeError("RoboDojo simulator failed") from self._failure
        result = super()._step(code)
        if self._failure is not None:
            raise RuntimeError("RoboDojo simulator failed") from self._failure
        return result

    def private_scoring_snapshot(self):
        from simulators.robodojo_scoring import snapshot

        return snapshot(self)

    def close(self):
        try:
            if self.native is not None:
                with robodojo_cwd():
                    self.native.close()
                self.native = None
        finally:
            self._temporary.cleanup()
            if self.app is not None:
                self.app.close()
                self.app = None
