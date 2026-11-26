# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compose the original ICL controller, CLI backend and Runner transport."""

from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path

from roborsi.agents.config import AgentConfig
from roborsi.planner.codex_cli_config import CodexCliConfig
from roborsi.utils.config import require_project_env

from .bridge import IclRoboRSIAgent
from .identity import verify_execution_identity
from .preparation import ROOT, DeploymentPaths, generate_fresh_config, verify_source
from .runtime_records import create_runtime

_ALLOWED = {
    "instance",
    "frozen_bundle",
    "benchmark_python",
    "agent_python",
    "environment_dir",
    "bubblewrap",
    "gpu",
    "physical_gpu",
    "vla_port",
    "execution_identity",
    "controller_python",
}


def prepare_case(case_dir: Path, *, config: AgentConfig) -> dict:
    """Prepare fresh authenticated inputs; never adopt author evaluation state."""
    document = config.document
    parameters = config.parameters
    if set(parameters) - _ALLOWED:
        raise ValueError("Unknown original ICL deployment parameter")
    required = _ALLOWED - {"gpu", "physical_gpu", "vla_port"}
    if required - set(parameters):
        raise ValueError("Original ICL requires explicit deployment paths and instance")
    options = document.get("agent", {})
    if (
        options.get("planner") != "codex_cli"
        or not isinstance(options.get("model"), str)
        or not options["model"].strip()
        or options.get("reasoning_effort") != "max"
        or options.get("max_turns") is not None
        or options.get("max_tokens") is not None
    ):
        raise ValueError("Original ICL requires an explicit CLI model and unchanged effort/budgets")
    runtime_model = os.environ.get("ROBORSI_CODEX_MODEL", "").strip()
    if not runtime_model or options["model"] != runtime_model:
        raise ValueError("ROBORSI_CODEX_MODEL must match the explicit prepared agent model")
    if (
        options.get("planner_timeout_s") != 0
        or options.get("action_timeout_s") is not None
    ):
        raise ValueError("Original controller must own normal model waiting deadlines")
    if (
        document.get("skills")
        or document.get("prompt_file")
        or "system_prompt" in options
        or document.get("runtime")
        or document.get("provider_capture")
    ):
        raise ValueError("Original ICL owns its prompt, tools, provider and context")
    identity = CodexCliConfig.from_dict(options.get("codex_cli", {}))
    if identity.tools != "external" or identity.expected_sha256 is None:
        raise ValueError(
            "Original ICL requires external tools and an actual binary SHA pin"
        )
    if (
        identity.cli_args
        or identity.config_overrides
        or identity.env
        or identity.env_vars
        or identity.provider_capture
    ):
        raise ValueError("Original CLI settings belong to the unchanged launcher")
    binary = Path(identity.executable).absolute()
    if not binary.is_file():
        raise ValueError("Missing original CLI executable")
    with binary.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"\x7fELF":
        raise ValueError("ICL executable must identify the actual native ELF binary")
    bubblewrap = Path(parameters["bubblewrap"]).absolute()
    if not bubblewrap.is_file():
        raise ValueError("Missing original bubblewrap executable")
    verify_source()
    base_deployment = DeploymentPaths(
        benchmark_python=Path(parameters["benchmark_python"]).absolute(),
        agent_python=Path(parameters["agent_python"]).absolute(),
        codex_binary=binary,
        environment_dir=Path(parameters["environment_dir"]).absolute(),
        gpu=parameters.get("gpu", 0),
        physical_gpu=parameters.get("physical_gpu", 0),
        vla_port=parameters.get("vla_port", 19802),
    )
    execution_identity = verify_execution_identity(
        identity,
        base_deployment,
        bubblewrap,
        Path(parameters["execution_identity"]),
        controller_python=Path(parameters["controller_python"]).absolute(),
    )
    # Original robot.sock must fit its unchanged 104-byte Unix-path bound.
    # Runtime records are checkpointed into the case after all writers stop.
    directory = create_runtime(case_dir)
    (directory / "execution-identity.json").write_text(
        json.dumps(execution_identity, indent=2)
    )
    source = directory / "runtime/source"
    shutil.copytree(
        ROOT / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    # Preserve immutable controller bytes. Only the deployment wrapper is new.
    benchmark_python = Path(parameters["benchmark_python"]).absolute()
    wrapper = directory / "benchmark-runtime"
    wrapper.write_text(
        "#!/bin/sh\nset -eu\nexec "
        + shlex.quote(str(benchmark_python))
        + " -u "
        + shlex.quote(str(source / "benchmark_bootstrap.py"))
        + ' "$@"\n',
    )
    wrapper.chmod(0o700)
    deployment = DeploymentPaths(
        benchmark_python=wrapper,
        agent_python=Path(parameters["agent_python"]).absolute(),
        codex_binary=binary,
        environment_dir=Path(parameters["environment_dir"]).absolute(),
        gpu=parameters.get("gpu", 0),
        physical_gpu=parameters.get("physical_gpu", 0),
        vla_port=parameters.get("vla_port", 19802),
    )
    cfg_path = generate_fresh_config(
        parameters["instance"],
        deployment,
        Path(parameters["frozen_bundle"]),
        directory / "preparation",
    )
    cfg = json.loads(cfg_path.read_text())
    cfg["model"] = options["model"]
    # Native service verifies and executes the immutable packaged adapter.
    # The controller process replaces only its own transport with worker_proxy.
    cfg["adapter"] = str(ROOT / "source/hybrid_adapter.py") + ":Adapter"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (directory / "preparation/import/config.json").write_text(json.dumps(cfg, indent=2))
    config_dir = source / "config"
    config_dir.mkdir()
    (config_dir / f"{parameters['instance']}.json").write_text(
        json.dumps(cfg, indent=2)
    )
    return {
        "source": str(source),
        "config": str(cfg_path),
        "runs": str(directory / "runs"),
        "receipt": str(directory / "controller-result.json"),
        "bubblewrap": str(bubblewrap),
        "cli_identity": {
            "executable": str(binary),
            "expected_version": identity.expected_version,
            "expected_sha256": identity.expected_sha256,
            "tools": "external",
        },
        "framework_source_root": str(Path(__file__).resolve().parents[4]),
        "experiment_root": str(directory),
        "case_dir": str(case_dir.absolute()),
        "execution_identity": execution_identity,
        "controller_python": str(Path(parameters["controller_python"]).absolute()),
    }


def make_agent(case_dir: Path, *, config: AgentConfig) -> IclRoboRSIAgent:
    """Build only the Agent; the Benchmark service owns the simulator."""
    request = prepare_case(case_dir, config=config)
    return IclRoboRSIAgent(
        request=request,
        directory=case_dir / "icl/controller",
        media_dir=Path(require_project_env("ROBORSI_MEDIA_DIR")),
    )


def _resume_runtime(case_dir):
    from .runtime_records import assert_runtime_owner

    case = Path(case_dir).absolute()
    reference = case / "icl/runtime-reference.json"
    if reference.is_symlink():
        raise ValueError("ICL case reference must not be linked")
    runtime = Path(json.loads(reference.read_text())["runtime"])
    assert_runtime_owner(runtime, case)
    return case, runtime


def _prepare_resume_locked(case, runtime, source_run, infrastructure_reason):
    """Validate this precise stopped attempt before original recovery mutates state."""
    import hashlib

    from .artifact_restore import copy_native_resume_records, restore_native_artifacts
    from .resume import _owned, prepare_resume
    from .runtime_records import _assert_stopped

    _assert_stopped(runtime)
    source_run = _owned(Path(source_run), runtime)
    if not source_run.is_relative_to(runtime / "runs"):
        raise ValueError("Resume source is outside this own episode")
    binding_path = _owned(source_run / "roborsi-controller.json", runtime)
    binding = json.loads(binding_path.read_text())
    if binding.get("source_run") != str(source_run) or binding.get("case_dir") != str(
        case
    ):
        raise ValueError("Controller source binding differs from this case")
    attempt = Path(binding["attempt_dir"]).absolute()
    if attempt != case and not attempt.is_relative_to(case / "attempts"):
        raise ValueError("Controller attempt is outside this case's attempts")
    restore_native_artifacts(attempt, binding["framework_episode_id"])
    receipt = attempt / "native/icl-original/worker-close.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("Native close/artifact receipt has not been collected")
    native = json.loads(receipt.read_text())
    if native.get("native_closed") is not True:
        raise ValueError("Native worker shutdown is not verified")
    instance = json.loads(
        (runtime / "preparation/PREPARATION_RECEIPT.json").read_text()
    )["instance"]
    config_path = runtime / "runtime/source/config" / f"{instance}.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if (
        native.get("episode_id") != binding.get("framework_episode_id")
        or native.get("native_config_sha256") != binding.get("native_config_sha256")
        or native.get("native_config_sha256") != config_sha
    ):
        raise ValueError(
            "Native shutdown receipt is stale or belongs to another attempt"
        )
    reconstruction_path = attempt / "native/icl-reconstruction.json"
    if reconstruction_path.is_symlink() or not reconstruction_path.is_file():
        raise ValueError("Native reconstruction manifest has not been collected")
    reconstruction = json.loads(reconstruction_path.read_text())
    closure_rows = [
        row for row in reconstruction["files"] if row["path"] == "worker-close.json"
    ]
    if (
        len(closure_rows) != 1
        or hashlib.sha256(receipt.read_bytes()).hexdigest() != closure_rows[0]["sha256"]
        or not closure_rows[0]["chunks"]
        or any(
            chunk["episode_id"] != binding["framework_episode_id"]
            for chunk in closure_rows[0]["chunks"]
        )
    ):
        raise ValueError("Native close receipt does not match collected artifacts")
    cfg = json.loads((runtime / "preparation/config.json").read_text())
    episode = cfg["test_episodes"][0]
    identity = native.get("identity", {})
    if (
        identity.get("task_id") != episode["task"]
        or identity.get("seed") != episode["layout_id"]
        or identity.get("native", {}).get("eval_seed") != episode["eval_seed"]
        or identity.get("native", {}).get("config") != episode["robot"]
        or identity.get("native", {}).get("reset_seed") != episode["reset_seed"]
    ):
        raise ValueError(
            "Collected native artifacts belong to another task/robot/reset seed"
        )
    copy_native_resume_records(
        attempt,
        binding["framework_episode_id"],
        source_run,
        experiment_root=runtime,
        case_dir=case,
    )
    return (
        prepare_resume(
            runtime / "runtime/source",
            instance,
            source_run,
            runtime / "resume-state.json",
            experiment_root=runtime,
            preparation_root=runtime / "preparation",
            infrastructure_reason=infrastructure_reason,
        ),
        binding,
        native,
        attempt,
    )


def prepare_case_resume(
    case_dir: Path, source_run: Path, *, infrastructure_reason: str | None = None
):
    """Prepare only a stopped own episode with matching attempt and artifact evidence."""
    from .runtime_records import runtime_lock

    case, runtime = _resume_runtime(case_dir)
    with runtime_lock(runtime, case):
        outcome, _, _, _ = _prepare_resume_locked(
            case, runtime, source_run, infrastructure_reason
        )
        return outcome


def build_resume_submission(
    case_dir: Path, source_run: Path, *, infrastructure_reason: str | None = None
) -> dict:
    """Build paired Agent/Benchmark inputs atomically under the same lifecycle lock.

    Requires the retained runtime to be readable by both services. Independent
    hosts must transfer and authenticate the chain before these paths are used.
    """
    import hashlib
    import uuid

    from roborsi.evaluation.messages import Task

    from .runtime_records import runtime_lock

    case, runtime = _resume_runtime(case_dir)
    with runtime_lock(runtime, case):
        outcome, binding, native, attempt = _prepare_resume_locked(
            case, runtime, source_run, infrastructure_reason
        )
        if not outcome.recovered:
            return {
                "accepted": False,
                "receipt": str(outcome.receipt_path),
                "state": outcome.state,
            }
        cfg = json.loads(outcome.config_path.read_text())
        offset = cfg["options"].get("replay", {}).get("control_step", 0)
        full_limit = (
            cfg["options"]["review_window_steps"] * cfg["options"]["review_max_windows"]
        )
        task = Task.from_dict(native["identity"]).model_copy(
            update={"max_steps": full_limit - offset}
        )
        request = json.loads(
            (attempt / "icl/controller/controller-request.json").read_text()
        )
        request["config"] = str(outcome.config_path)
        request["receipt"] = str(
            runtime / "controller-results" / f"resume-{uuid.uuid4().hex}.json"
        )
        Path(request["receipt"]).parent.mkdir(exist_ok=True)
        submission = {
            "accepted": True,
            "task": task.as_dict(),
            "step_offset": offset,
            "remaining_seconds": cfg["episode_seconds"],
            "native_config": str(outcome.config_path),
            "native_config_sha256": hashlib.sha256(
                outcome.config_path.read_bytes()
            ).hexdigest(),
            "controller_request": request,
            "receipt": str(outcome.receipt_path),
            "source_run": str(source_run),
            "shared_runtime_required": True,
            "author_evaluation_state_loaded": False,
        }
        output = runtime / "resume-submissions"
        output.mkdir(exist_ok=True)
        (output / f"resume-{uuid.uuid4().hex}.json").write_text(
            json.dumps(submission, indent=2)
        )
        return submission


def make_resumed_agent(
    attempt_dir: Path, submission: dict, *, media_dir: Path
) -> IclRoboRSIAgent:
    """Create a distinct Runner attempt for the same authenticated retained episode."""
    import hashlib

    from .runtime_records import assert_runtime_owner

    if (
        submission.get("accepted") is not True
        or submission.get("author_evaluation_state_loaded") is not False
    ):
        raise ValueError("Original recovery did not accept an own-episode continuation")
    request = submission["controller_request"]
    runtime, case = Path(request["experiment_root"]), Path(request["case_dir"])
    assert_runtime_owner(runtime, case)
    attempt = attempt_dir.absolute()
    if (
        not attempt.is_relative_to(case / "attempts")
        or (attempt / "icl/controller").exists()
    ):
        raise ValueError("Resume needs a new output below this case's attempts")
    cfg_path = Path(request["config"]).absolute()
    if not cfg_path.is_relative_to(runtime) or cfg_path.is_symlink():
        raise ValueError("Resume config is outside this retained runtime")
    if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != submission.get(
        "native_config_sha256"
    ):
        raise ValueError("Prepared resume configuration changed")
    cfg = json.loads(cfg_path.read_text())
    recovery = cfg.get("local_integration", {}).get("recovery", {})
    if (
        not recovery.get("same_episode")
        or recovery.get("source_run") != submission["source_run"]
    ):
        raise ValueError(
            "Resume config is not the original accepted same-episode recovery"
        )
    return IclRoboRSIAgent(
        request=request, directory=attempt / "icl/controller", media_dir=media_dir
    )
