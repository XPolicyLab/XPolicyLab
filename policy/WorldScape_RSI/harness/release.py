"""Freeze development choices as data; inference cannot rewrite a release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = "r2si.inner-agent-wam-skills.v2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contained(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Release path escapes its root")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Release asset must be a regular file: " + name)
    return path


WAM_IDS = {"worldscape-wam-memory", "worldscape-wam-nomemory"}


def validate(data: dict) -> None:
    if data.get("schema") != SCHEMA or data.get("mode") != "inner_agent_wam_skills":
        raise ValueError("Expected a common inner Agent with WAM Skills")
    if any(key in data for key in ("routes", "backends", "selection_provenance")):
        raise ValueError("Task-to-executor routing is not permitted")
    if data.get("official_single_entry_eligible") is not False:
        raise ValueError("Eligibility remains unverified until a protocol audit")
    if data.get("supported") != {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint"}:
        raise ValueError("Only the verified ARX X5 joint interface is supported")
    agent = data.get("inner_agent", {})
    if (agent.get("kind") != "agent" or not agent.get("model")
            or agent.get("effort") != "high" or agent.get("memory") != "full_episode"
            or not agent.get("api_base_env") or not agent.get("api_key_env")):
        raise ValueError("The inner Agent model and memory contract must be frozen")
    skills = data.get("wam_skills", {})
    if set(skills) != WAM_IDS:
        raise ValueError("Both and only the WSP memory/no-memory Skills are required")
    for name, spec in skills.items():
        if (spec.get("profile") != {"worldscape-wam-memory": "mem20000", "worldscape-wam-nomemory": "checkpoint30000"}[name]
                or spec.get("kind") != "checkpoint" or spec.get("policy_name") != "WorldScape_Policy_2"
                or spec.get("observation_contract") != "wsp_explicit_commit"
                or not spec.get("checkpoint_env") or not spec.get("checkpoint_files")
                or "model.py" not in spec.get("source_files", {})):
            raise ValueError("WAM Skills require hash-pinned WSP checkpoints and explicit commit")
        cfg = spec.get("config", {})
        if (cfg.get("runtime_profile") != "robodojo" or cfg.get("eval_batch") is not False
                or cfg.get("action_type") != "joint" or cfg.get("multi_gpu", {}).get("enabled")
                or cfg.get("interactive_qwen", {}).get("enabled")
                or cfg.get("press_by_number_prompt", {}).get("enabled")
                or any(k in cfg for k in ("task_name", "routes", "checkpoint_path"))):
            raise ValueError("WAM configuration must be task-independent and inference-only")
        if (cfg.get("short_window_for_non_memory_tasks") is not False
                or cfg.get("memory_window_chunks") is not None
                or cfg.get("ordinary_window_chunks") is not None
                or "gripper_discretization" in cfg):
            raise ValueError("Use task-independent WSP history and checkpoint-owned transforms")
        expected = ({"use_observation_history": False, "use_causal_memory": False,
                     "reset_independent_blocks": True, "wam_history_mode": "periodic",
                     "observation_history_segments": 1}
                    if name == "worldscape-wam-nomemory" else
                    {"use_observation_history": True, "use_causal_memory": True,
                     "reset_independent_blocks": False, "wam_history_mode": "rolling",
                     "observation_history_segments": 8})
        if any(type(cfg.get(key)) is not type(value) or cfg[key] != value
               for key, value in expected.items()):
            raise ValueError("WAM memory semantics differ from the declared Skill variant")
        for group in ("checkpoint_files", "source_files"):
            for relative, value in spec[group].items():
                if (Path(relative).is_absolute() or ".." in Path(relative).parts
                        or not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError("Invalid WAM resource manifest")


class FrozenRelease:
    def __init__(self, path: Path, expected_digest: str | None = None):
        self.path = path.resolve()
        self.root = self.path.parent
        self.raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        if expected_digest is not None and self.sha256 != expected_digest:
            raise ValueError("Release digest differs from deployment configuration")
        self.data = json.loads(self.raw)
        validate(self.data)
        if not self.data.get("files"):
            raise ValueError("Release has no code/Skill integrity manifest")
        self.verify()

    def verify(self) -> None:
        if self.path.read_bytes() != self.raw:
            raise RuntimeError("Release changed while the Harness was running")
        for name, expected in self.data["files"].items():
            if digest(contained(self.root, name)) != expected:
                raise RuntimeError("Frozen code or Skill changed: " + name)
        for name, expected in self.data.get("integration_files", {}).items():
            if digest(contained(self.root.parent, name)) != expected:
                raise RuntimeError("Frozen policy integration changed: " + name)

    def agent_spec(self) -> dict:
        spec = json.loads(json.dumps(self.data["inner_agent"]))
        spec["wam_skills"] = json.loads(json.dumps(self.data["wam_skills"]))
        return spec


def freeze(root: Path = ROOT, destination: str = "release.json") -> Path:
    target = root / destination
    if target.exists():
        raise FileExistsError("Create a new release directory instead of replacing a release")
    data = {
        "schema": SCHEMA,
        "mode": "inner_agent_wam_skills",
        "official_single_entry_eligible": False,
        "supported": {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint"},
        "inner_agent": json.loads((root / "configs/inner_agent.json").read_text()),
        "wam_skills": json.loads((root / "configs/wam_skills.json").read_text()),
        "files": {},
        "integration_files": {},
    }
    validate(data)
    for path in sorted(root.rglob("*")):
        if (path.is_file() and path != target and "__pycache__" not in path.parts
                and "tests" not in path.relative_to(root).parts
                and (path.suffix in {".py", ".json", ".md", ".sh", ".yml", ".yaml", ".txt"}
                     or "runtime_support" in path.relative_to(root).parts)):
            name = path.relative_to(root).as_posix()
            data["files"][name] = digest(contained(root, name))
    if root.name == "harness":
        for name in ("model.py", "deploy.py", "server.py", "deploy.yml", "__init__.py",
                     "install.sh", "eval.sh", "setup_eval_policy_server.sh", "setup_eval_env_client.sh",
                     "requirements.txt"):
            data["integration_files"][name] = digest(contained(root.parent, name))
    with target.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    FrozenRelease(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = ROOT / "release.json"
    if args.verify:
        release = FrozenRelease(path)
    else:
        path = freeze()
        release = FrozenRelease(path)
    print(json.dumps({"release_sha256": release.sha256, "wam_skills": sorted(release.data["wam_skills"]),
                      "official_single_entry_eligible": False}))
