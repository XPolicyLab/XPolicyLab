"""Freeze development choices as data; inference cannot rewrite a release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
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


WAM_CONFIG_KEYS = frozenset(['action_horizon', 'action_type', 'bench_name', 'constrain_action_padding', 'default_instruction', 'device', 'diffusion_view_layout', 'env_cfg_type', 'eval_batch', 'input_color_order', 'mode', 'multi_gpu', 'observation_history_segments', 'observation_interval', 'policy_image_interpolation', 'policy_name', 'replan_steps', 'reset_independent_blocks', 'runtime_profile', 'seed', 'single_window_reference', 't5_prompt_template', 'use_causal_memory', 'use_observation_history', 'validate_checkpoint_artifacts', 'wam_history_mode'])


def validate(data: dict) -> None:
    if data.get("schema") != SCHEMA or data.get("mode") != "inner_agent_wam_skills":
        raise ValueError("Expected a common inner Agent with WAM Skills")
    if set(data) - {"schema", "mode", "official_single_entry_eligible", "supported", "inner_agent",
                    "wam_skills", "files", "integration_files"}:
        raise ValueError("Unsupported release fields")
    if data.get("official_single_entry_eligible") is not False:
        raise ValueError("Eligibility remains unverified until a protocol audit")
    if data.get("supported") != {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint"}:
        raise ValueError("Only the verified ARX X5 joint interface is supported")
    agent = data.get("inner_agent", {})
    if (agent.get("kind") != "agent" or not agent.get("model")
            or agent.get("effort") != "high" or agent.get("memory") != "full_episode"
            or not agent.get("api_base_env") or not agent.get("api_key_env")):
        raise ValueError("The inner Agent model and memory contract must be frozen")
    if agent.get('wam_command_profile', 'native_planner_budget') not in {
            'native_planner_budget', 'actuator_velocity_budget'}:
        raise ValueError('Unknown frozen WAM command profile')
    skills = data.get("wam_skills", {})
    if set(skills) != WAM_IDS:
        raise ValueError("Both and only the WSP memory/no-memory Skills are required")
    for name, spec in skills.items():
        if (spec.get("profile") != {"worldscape-wam-memory": "memory", "worldscape-wam-nomemory": "flash"}[name]
                or spec.get("kind") != "checkpoint" or spec.get("policy_name") != "WorldScape_RSI"
                or spec.get("observation_contract") != "wsp_explicit_commit"
                or not spec.get("checkpoint_env") or not spec.get("checkpoint_files")
                or "wam_model.py" not in spec.get("source_files", {})):
            raise ValueError("WAM Skills require hash-pinned WSP checkpoints and explicit commit")
        cfg = spec.get("config", {})
        if (cfg.get("runtime_profile") != "robodojo" or cfg.get("eval_batch") is not False
                or cfg.get("action_type") != "joint" or cfg.get("multi_gpu", {}).get("enabled")
                or set(cfg) - WAM_CONFIG_KEYS):
            raise ValueError("WAM configuration must be task-independent and inference-only")
        # Both published checkpoints use rolling inference. Flash's checkpoint
        # capacity is one chunk, which limits its visual input to the latest frame.
        expected = {"use_observation_history": True, "use_causal_memory": True,
                    "reset_independent_blocks": False, "wam_history_mode": "rolling",
                    "observation_history_segments": 8}
        if any(type(cfg.get(key)) is not type(value) or cfg[key] != value
               for key, value in expected.items()):
            raise ValueError("WAM memory semantics differ from the declared Skill variant")
        interval, steps = cfg.get("observation_interval"), cfg.get("replan_steps")
        if (type(interval) is not int or interval <= 0 or type(steps) is not int or steps <= 0
                or steps % interval or steps // interval != cfg["observation_history_segments"]):
            raise ValueError("WAM sampling geometry is inconsistent")
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

    @staticmethod
    def _signature(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("Frozen release cannot redirect through a symlink")
        if stat.S_ISDIR(info.st_mode):
            # Import caches alter directory timestamps, size and link count.
            # Compare directory identity/permissions and non-cache membership
            # instead; tracked source files retain the full metadata signature.
            members = []
            for child in path.iterdir():
                if child.name == "__pycache__":
                    cache_info = child.lstat()
                    if not stat.S_ISDIR(cache_info.st_mode):
                        raise RuntimeError("Python cache must be a real directory")
                    continue
                members.append(child.name)
            return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                    info.st_gid, tuple(sorted(members)))
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns, info.st_uid, info.st_gid, info.st_nlink)

    def check_unchanged(self):
        # Full content validation at construction and every reset. Between
        # actions, fail closed on file or ancestor metadata changes. The sealed
        # release is single-writer; this is not a defence against a privileged
        # filesystem administrator forging timestamps. Use full mode on mounts
        # without reliable ctime/mtime or when that trust model is unsuitable.
        if os.environ.get("WRSI_INTEGRITY_MODE", "metadata") == "full":
            return self.verify()
        for path, expected in self._signatures.items():
            if self._signature(path) != expected:
                raise RuntimeError("Frozen release changed: " + str(path))

    def verify(self) -> None:
        if self.path.read_bytes() != self.raw:
            raise RuntimeError("Release changed while the Harness was running")
        verified = {self.path: self._signature(self.path)}
        paths = [self.path]
        for name, expected in self.data["files"].items():
            asset = contained(self.root, name)
            before = self._signature(asset)
            if digest(asset) != expected or self._signature(asset) != before:
                raise RuntimeError("Frozen code or Skill changed: " + name)
            paths.append(self.root / name)
            verified[asset] = before
        for name, expected in self.data.get("integration_files", {}).items():
            asset = contained(self.root.parent, name)
            before = self._signature(asset)
            if digest(asset) != expected or self._signature(asset) != before:
                raise RuntimeError("Frozen policy integration changed: " + name)
            paths.append(self.root.parent / name)
            verified[asset] = before
        # Ancestors prevent replacing a directory with a redirected tree.
        for path in list(paths):
            for parent in path.parents:
                if parent == self.root.parent.parent:
                    break
                paths.append(parent)
        self._signatures = {path: verified[path] if path in verified else self._signature(path)
                            for path in set(paths)}
        for path, expected in self._signatures.items():
            if self._signature(path) != expected:
                raise RuntimeError("Release changed during verification")

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
                     "requirements.txt", "requirements-runtime.txt", "preflight.py", "maintain_manifest.py"):
            data["integration_files"][name] = digest(contained(root.parent, name))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
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
