"""Build and run one episode from a JSON config.

{
  "backend": {"kind": "robodojo", "task": "stack_blocks", "seed": 0, "depth": "none", "calibration": true},
  "perception": {"depth": "service" | "native" | "none", "sam3_url": null, "depth_url": null},
  "models": {"task": {...}, "local": {...}},          # ModelConfig fields, provider openai_compat
  "limits": {"max_task_decisions": 150, ...},
  "skills_dir": null,
  "brief": "robodojo",                               # scoring notes for the task, a file path, or null
  "video_stride": 2                                  # 0 disables video
}
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from knowin_agent.agents.controller import EpisodeController, EpisodeLimits
from knowin_agent.agents.llm import build_model
from knowin_agent.core.memory import TaskContext
from knowin_agent.core.recorder import RunRecorder
from knowin_agent.runtime.api import RobotRuntime
from knowin_agent.skills.library import SkillLibrary


def robodojo_brief(task: str) -> str:
    base = resources.files("knowin_agent.assets").joinpath("robodojo", "scoring")
    parts = [base.joinpath("_common.md").read_text(encoding="utf-8")]
    name = task[:-len("_random")] if task.endswith("_random") else task
    note = base.joinpath(f"{name}.md")
    if note.is_file():
        parts.append(note.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def build_backend(spec: dict, *, frame_sink=None):
    kind = spec.get("kind", "robodojo")
    if kind == "robodojo":
        from knowin_agent.backends.robodojo import RoboDojoBackend

        return RoboDojoBackend(spec["task"], int(spec.get("seed", 0)), depth=spec.get("depth", "none"),
                               calibration=spec.get("calibration", True),
                               official_action_limit=bool(spec.get("official_action_limit", True)),
                               robodojo_root=spec.get("robodojo_root"), kit_args=spec.get("kit_args"),
                               frame_sink=frame_sink)
    if kind == "mock":
        from knowin_agent.backends.mock import demo_world

        return demo_world(ticks_limit=spec.get("ticks_limit", 3000))
    raise ValueError(f"unknown backend kind {kind!r}")


def build_perceiver(spec: dict, backend):
    from knowin_agent.backends.mock import MockBackend, MockPerceiver

    if isinstance(backend, MockBackend):
        return MockPerceiver(backend)  # the mock world has ground-truth perception only
    mode = spec.get("depth", "service")
    if mode == "none":
        return None
    from knowin_agent.perception.perceiver import Perceiver
    from knowin_agent.perception.services import CachedDepth, CachedSegmenter, HttpPointDepth, HttpSegmenter, NativeDepth

    segmenter = CachedSegmenter(HttpSegmenter(spec.get("sam3_url")))
    depth = NativeDepth() if mode == "native" else HttpPointDepth(spec.get("depth_url"))
    return Perceiver(segmenter, CachedDepth(depth), table_z=backend.profile.table_z)


def build_limits(values: Optional[dict]) -> EpisodeLimits:
    values = dict(values or {})
    known = {f.name for f in fields(EpisodeLimits)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown limits {sorted(unknown)}")
    return EpisodeLimits(**values)


def build_stack(config: dict, *, backend=None, frame_sink=None):
    """Backend, runtime and skill library for a config (also used by smoke tests without models)."""
    backend_spec = config.get("backend", {})
    backend = backend or build_backend(backend_spec, frame_sink=frame_sink)
    perceiver = build_perceiver(config.get("perception", {}), backend)
    runtime = RobotRuntime(backend, context=TaskContext(instruction=backend.instruction()), perceiver=perceiver)
    library = SkillLibrary()
    if config.get("skills_dir"):
        library.load_dir(Path(config["skills_dir"]))
    return backend, runtime, library


def run_episode(config: dict, run_dir: Path, *, task_model=None, local_model=None, backend=None) -> dict:
    """One episode with API-model agents (OpenAI-compatible); switches, workspace and program mode applied."""
    from knowin_agent.assembly import assemble

    run_dir = Path(run_dir)
    recorder = RunRecorder(run_dir)
    recorder.write_json("config.json", config)
    stride = int(config.get("video_stride", 2))
    frames = None
    if stride > 0:
        from knowin_agent.backends.robodojo.video import FrameRecorder

        frames = FrameRecorder(stride=stride)
    started = time.time()
    kit = assemble(config, run_dir, backend=backend, frame_sink=frames, recorder=recorder)
    recorder.record("episode", phase="backend_ready", seconds=round(time.time() - started, 1),
                    instruction=kit.backend.instruction())
    try:
        models = kit.config.get("models", {})
        task_model = task_model or build_model(models["task"])
        local_model = local_model or (build_model(models["local"]) if models.get("local") else task_model)
        controller = EpisodeController(kit.runtime, kit.library, task_model=task_model, local_model=local_model,
                                       recorder=recorder, limits=build_limits(kit.config.get("limits")),
                                       task_brief=kit.brief_text, image_side=int(kit.config.get("image_side", 448)),
                                       kit=kit)
        result = controller.run()
        return finish_run(kit, run_dir, result)
    finally:
        if frames is not None and frames.frames:
            frames.write(run_dir / "video", fps=kit.backend.profile.control_hz)
        kit.backend.close()


def finish_run(kit, run_dir: Path, result: dict) -> dict:
    """Freeze practice knowledge and write the run manifest (append-only)."""
    from knowin_agent.experiments.manifest import collect, write_manifest

    run_dir = Path(run_dir)
    if kit.workspace is not None and kit.phase == "practice":
        from knowin_agent.workspace import freeze

        manifest = freeze(kit.workspace, run_dir / "frozen", source_run=run_dir.name,
                          library_version=kit.workspace.shared_version)
        result["frozen_dir"] = str(Path(manifest).parent)
    result["executions"] = kit.executions
    evaluation = result.get("evaluation_host_only") or {}
    stats = collect(run_dir)
    try:
        backend_spec = kit.config.get("backend", {})
        write_manifest(run_dir,
                       identity={"run_id": run_dir.name, "task": backend_spec.get("task") or backend_spec.get("kind"),
                                 "phase": kit.phase,
                                 **(kit.config.get("identity") or {})},
                       switches=kit.switches, profile=kit.config.get("profile"),
                       library_version=kit.workspace.shared_version if kit.workspace else None,
                       knowledge_source=kit.config.get("knowledge_from"), brief=kit.brief_manifest,
                       scene={"seed": backend_spec.get("seed"), "bank": evaluation.get("bank"),
                              "ordinal": evaluation.get("ordinal"), "split": kit.config.get("split")},
                       model=kit.config.get("agent") or kit.config.get("models"),
                       results={"score": evaluation.get("score"), "success": evaluation.get("success")},
                       costs={**stats.get("costs", {}), "executions": kit.executions,
                              "wall_seconds": result.get("wall_seconds")},
                       process={**stats.get("process", {}), "stop_reason": result.get("stop_reason")})
    except FileExistsError:
        pass
    except Exception as exc:  # noqa: BLE001 - a manifest problem must not lose the episode result
        result["manifest_error"] = f"{type(exc).__name__}: {exc}"
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    return result


def load_config(path: Path, overrides: Optional[dict[str, Any]] = None) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        node = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return config
