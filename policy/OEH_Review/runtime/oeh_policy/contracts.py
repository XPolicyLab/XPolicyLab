"""Frozen runtime configuration and policy-only observation contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .profiles import REVIEW_BENCHMARKS, check_embodiment, normalize_benchmark

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")
PHASES = ("grasp", "verify_grasp", "carry", "place", "verify_place", "retreat", "home")
CAPABILITY = "oeh_manipulate"


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("expected finite JSON data")


def encoded(value: Any) -> bytes:
    return (json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def project_state(state: Mapping) -> dict:
    result = {}
    for key in STATE:
        array = np.asarray(state[key])
        size = 7 if key.endswith("_pose") else 1
        if array.shape != (size,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"invalid robot state {key}")
        result[key] = array.copy()
    return result


def project_observation(obs: Mapping) -> dict:
    """Copy only the pinned OpenWAM RGB/EEF contract, including nested fields.

    No simulator object, evaluator state, depth, calibration or additional robot
    state passes this boundary. The policy cannot retain aliases into caller data.
    """
    result = {"state": project_state(obs["state"]), "vision": {}}
    for camera in CAMERAS:
        image = np.asarray(obs["vision"][camera]["color"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
            raise ValueError(f"invalid RGB image {camera}")
        result["vision"][camera] = {"color": image.copy()}
    for key in ("instruction", "task_instruction"):
        if key in obs:
            if not isinstance(obs[key], str) or not obs[key].strip():
                raise ValueError(f"invalid {key}")
            result[key] = obs[key]
    if not ("instruction" in result or "task_instruction" in result):
        raise ValueError("task instruction required")
    if "env_idx" in obs:
        if isinstance(obs["env_idx"], bool) or not isinstance(obs["env_idx"], (int, np.integer)):
            raise ValueError("invalid environment index")
        result["env_idx"] = int(obs["env_idx"])
    return result


def observation_digest(obs: Mapping) -> str:
    # Hash raw arrays with shape and dtype; do not serialize megapixels as JSON.
    def walk(value):
        if isinstance(value, np.ndarray):
            return {"sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}
        return plain(value)
    return digest(walk(obs))


def validate_config(value: Mapping) -> dict:
    cfg = copy.deepcopy(dict(value))
    required = {"schema", "mode", "kernel", "repository", "output", "model_id",
                "checkpoint_sha256", "normalization_sha256", "identity_files",
                "max_chunk_steps", "native_step_limit", "wall_seconds", "decision_seconds",
                "action_limits", "gripper_state_semantics", "embodiment"}
    if required - cfg.keys():
        raise ValueError(f"missing configuration fields: {sorted(required - cfg.keys())}")
    if cfg["schema"] != "oeh.policy.v1" or cfg["mode"] not in {"fast", "slow", "hybrid", "review", "review-stmem"}:
        raise ValueError("unsupported policy configuration")
    check_embodiment(cfg["embodiment"], cfg.get("benchmark", "robodojo"))
    if cfg.get("data_kind", "simulation") not in {"simulation", "synthetic"}:
        raise ValueError("this implementation supports simulation and synthetic fixtures only")
    if cfg["mode"] == "fast" and cfg.get("brain"):
        raise ValueError("fast mode cannot configure an additional model")
    if cfg["gripper_state_semantics"] != "previous_command":
        raise ValueError("initial gripper ledger requires declared previous-command state")
    for key in ("max_chunk_steps", "native_step_limit"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"positive integer {key} required")
    # first_decision_seconds (optional, default decision_seconds) bounds the episode's first decision, which
    # pays the model's warm-up; absent, frozen configurations keep their hash and behaviour.
    for key in ("wall_seconds", "decision_seconds", "first_decision_seconds"):
        if key not in cfg and key == "first_decision_seconds":
            continue
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < math.inf:
            raise ValueError(f"finite positive {key} required")
    for key in ("checkpoint_sha256", "normalization_sha256"):
        if not isinstance(cfg[key], str) or len(cfg[key]) != 64 or any(c not in "0123456789abcdef" for c in cfg[key]):
            raise ValueError(f"frozen {key} required")
    if not cfg["identity_files"]:
        raise ValueError("source, checkpoint and normalization identity files required")
    for path, expected in cfg["identity_files"].items():
        if not Path(path).is_absolute() or file_digest(Path(path)) != expected:
            raise ValueError(f"runtime identity mismatch: {path}")
    limits = cfg["action_limits"]
    if limits.get("frame") != "env_relative_world" or limits.get("quaternion") != "wxyz":
        raise ValueError("OpenWAM native EE frame and quaternion contract required")
    for arm in ("left", "right"):
        bounds = np.asarray(limits[arm + "_xyz_bounds"], dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("finite robot-derived workspace bounds required")
    if cfg["mode"] in {"slow", "hybrid"}:
        brain = cfg.get("brain", {})
        if not all(brain.get(k) for k in ("provider", "model", "providers")):
            raise ValueError("slow mode requires a configured internal model")
        unbounded = cfg["mode"] == "slow" and brain.get("agent_termination") is True and brain.get("pose_control") is True
        if type(brain.get("max_calls")) is not int or brain["max_calls"] < (0 if unbounded else 1):
            raise ValueError("slow mode requires a model call budget")
    if cfg["mode"] == "hybrid":
        from .escalation import validate_escalation
        cfg["escalation"] = validate_escalation(cfg.get("escalation"))
    elif "escalation" in cfg:
        raise ValueError("only hybrid mode configures escalation")
    if cfg["mode"] in {"review", "review-stmem"}:
        if cfg.get("brain"):
            raise ValueError("review mode configures its teacher under review.teacher, not brain")
        check_review_benchmark(cfg.get("benchmark", "robodojo"), cfg.get("review"))
        cfg["review"] = validate_review(cfg.get("review"))
        if cfg["review"].get("llm_token_budget") is not None:
            from .llm_budget import resolve as resolve_llm_budget
            cfg["review"]["llm_token_budget"]=resolve_llm_budget(
                cfg["review"]["llm_token_budget"],cfg["native_step_limit"])
    elif "review" in cfg:
        raise ValueError("only review mode configures review")
    if cfg["mode"] == "review-stmem":
        from .stmem_runtime import validate_stmem
        cfg["stmem"] = validate_stmem(cfg)
    elif "stmem" in cfg:
        raise ValueError("stmem requires review-stmem mode")
    if cfg["mode"]=="review-stmem":
        cfg["experiment_variant"]="review-stmem-semmem" if "semmem" in cfg else "review-stmem"
    if "semmem" in cfg:
        from .task_tree import validate_config as validate_semmem
        cfg["semmem"] = validate_semmem(cfg)
    return cfg


# The review-block keys a `designs:` entry's mechanisms set (oeh_policy.mechanisms; condition-registry.md). Absent,
# nothing is added: the review block and its hash stay as they were. Each key is validated by the module that
# realizes it (segment: takeover; digest: review; notes_store: notes_store), and refused here until one does.
DESIGN_REVIEW_KEYS = ("segment", "digest", "handback", "notes_store")
REVIEW_DEFAULTS = {"call_seconds": 90, "hold_reserve_s": 12, "first_call_extra_seconds": 0,
                   "teacher_max_wait_s": 900, "turn_seconds": 1800, "empty_candidate_holds": 3,
                   "max_decisions": 0, "control_hz": 25, "task": None, "limits": {}}
TEACHER_DEFAULTS = {"api": "openai_responses", "base_url": "https://api.openai.com/v1", "request_timeout_s": 150,
                    "image_max_edge": 480, "max_attempts": 4, "retry_delays_s": [5, 15, 30], "max_invalid": 6,
                    "prune_above_tokens": 180000, "context_images": 6, "max_output_tokens": None,
                    "prices_usd_per_million": None}
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}  # output_config.effort with adaptive thinking
TEACHER_APIS = {"openai_responses", "codex_app_server", "anthropic_messages", "claude_code", "openai_chat"}
# Transports billed per token: their usage rows carry a list-price estimate (prices_usd_per_million), which the review
# queue's API caps add up. Without prices every estimate is empty and a cap reads $0. The Codex and Claude Code
# transports run on the user's subscription quota: no prices.
PAID_TEACHER_APIS = {"openai_responses", "anthropic_messages", "openai_chat"}
# Where each transport's credentials live (a path, never the credentials): an API key file, or a file listing login
# directories (Codex's CODEX_HOME, Claude Code's CLAUDE_CONFIG_DIR).
TEACHER_CREDENTIALS = {"openai_responses": "api_key_file", "codex_app_server": "codex_accounts_file",
                       "anthropic_messages": "api_key_file", "claude_code": "claude_accounts_file",
                       "openai_chat": "api_key_file"}
CLAUDE_APIS = {"anthropic_messages", "claude_code"}
# Where the teacher's model loop runs: in this process (`python`, oeh_policy.teacher.TeacherBrain) or in the kernel's
# slow-brain module (`brain`, oeh-brain's `brain-teacher` lane, which asks this process for state and validation).
# Absent, the channel is python and the configuration (and its hash) is the one the formal passes froze.
TEACHER_CHANNELS = ("python", "brain")
# The transports the brain channel implements so far.
BRAIN_CHANNEL_APIS = {"codex_app_server"}
TEACHER_BASE_URLS = {"anthropic_messages": "https://api.anthropic.com"}

# review.context: the declared context manifest (docs/agent-icl-study/declared-factor-mode.md §1). Absent, the review
# block, its hash and the teacher's bytes are R0's. Present, every switch is frozen at an explicit value, each
# switch's "off" value being R0's realization (the bytes the teacher reads under it are R0's), so the manifest with
# every switch off is R0 and a lock may declare it (R0, R0b). Namespaces follow the study's taxonomy; memory and
# history sit under state. Each switch: (off value, kind), kind being a tuple of allowed values, bool, ("int", low,
# high), "file" (a path under the benchmark's prompt directory that must exist) or "history" (the sub-block).
CONTEXT_SWITCHES = {
    "task": {"note": ("brief", ("none", "none_process", "brief", "recipe")),
             "thresholds": (True, bool), "step_budget": (True, bool), "recipe_file": (None, "file"),
             # G8 (experiment-plan-round1.md §3): does showing the decision budget (`decisions_left`, next to
             # `remaining_steps`) in every packet change how the LLM component bounds its own loops? Off: R0's packet.
             "decision_budget": (False, bool)},
    "state": {"embodiment": ("none", ("none", "rich")), "embodiment_file": (None, "file"),
              "episode_record": (False, bool), "placebo_tokens": (0, ("int", 0, 4000)),
              "placebo_file": (None, "file"),
              "history": ({"policy": "transport", "recent": None, "max_anchors": None}, "history")},
    "behavior": {"lessons_file": (None, "file"), "examples_file": (None, "file"), "examples_k": (0, ("int", 0, 16))},
    "supervisor": {"arrival_check": (False, bool),
                   "channel": (None, (None, "anticipatory", "reactive", "shadow"))},
}
HISTORY_POLICIES = ("transport", "recency", "anchored")
HISTORY_BOUNDS = {"recent": (1, 32), "max_anchors": (0, 32)}
# Switches whose value may leave its off value, as "namespace.key": the ones this commit's runtime realizes (changes
# the teacher's bytes where the specification says and nowhere else). A switch that is declared on but not here is
# refused, so an arm never runs R0's bytes under its own name. R0 and R0b compile; each entry names its realization.
CONTEXT_REALIZED: frozenset = frozenset({
    "task.decision_budget",  # G8: runtime._review_wake writes decisions_left (tests/test_decisions_left.py)
    # G3 (oeh_policy.review_context; byte diffs per level in tests/test_review_context.py):
    "task.note",  # task_context_for (none / none_process / recipe), note none's SKILL.md removal
    "task.recipe_file",  # the recipe section in task_context; its sha in prompt_identity (test_review_context.py)
    "state.embodiment",  # the "# Embodiment record" insertion after teacher_context.md (test_review_context.py)
    "state.embodiment_file",  # the inserted file; its sha in prompt_identity (test_review_context.py)
    "behavior.lessons_file",  # the "# Lessons from earlier episodes" read path, last; its sha (test_review_context.py)
})
# Switches whose on value can only mean something with another setting (declared-factor-mode.md §1.3 rule 5).
CONTEXT_FILE_PAIRS = (("task", "note", "recipe", "recipe_file"), ("state", "embodiment", "rich", "embodiment_file"))

# The teacher packet's oracle boundary (declared-factor-mode.md §4): keys no packet may carry at any depth, and the
# top-level field sets. A field set is a name composed from v1 (R0's packet) and what the review block switches on;
# it is a field list, not a version number, and identity.json carries it.
PACKET_FORBIDDEN_KEYS = frozenset({"layout", "layout_id", "seed", "seed_id", "score", "reward", "success",
                                   "native_success", "object", "pose_truth", "env_idx", "depth", "segmentation"})
# task_started (PR #42, develop d7c8c549 lineage): every review packet carries whether a non-hold task action has
# executed; it entered the base packet with that merge (2026-10-20), so the v1 field set names it.
PACKET_FIELDS_V1 = ("request_id", "decision", "observation_id", "candidate_id", "native_step", "max_episode_steps",
                    "remaining_steps", "sim_time_s", "instruction", "current_eef", "episode_start_eef", "home_offset",
                    "student_proposal", "previous_result", "counters", "images", "task_started")
PACKET_OPTIONAL_V1 = ("task_context",)  # decision 0 only
# "+name": the top-level keys the addition brings (required when present in the set; arrival is nested in
# previous_result and adds none); "-name": the v1 keys the removal drops.
PACKET_ADDITIONS = {"guard": ("review_guard",), "takeover": ("takeover",), "arrival": (), "record": ("episode_record",),
                    "padding": ("padding",), "decisions": ("decisions_left",), "digest": ("interval_digest",),
                    "notes": ("task_notes",), "probe": ("exploration_probe",)}
PACKET_REMOVALS = {"step": ("max_episode_steps", "remaining_steps")}
# Additions whose keys are allowed but not required on every packet: task_notes is the first packet's (G7), like
# task_context.
PACKET_OPTIONAL_ADDITIONS = frozenset({"notes"})


def check_review_benchmark(benchmark: Any, review: Mapping | None = None) -> str:
    """The benchmark a review-mode configuration runs on: one whose contract implements review mode
    (profiles.REVIEW_BENCHMARKS, the registry's ``review``), normalized. Every contract's validate_config calls this
    before validate_review. Off RoboDojo it also refuses the review settings only RoboDojo's review implements: the
    review guard (its rules and thresholds are calibrated on RoboDojo's arm-return check, OpenWAM-RoboDojo's grip band
    and its workspace bounds), the on_stall schedule (oeh_policy.takeover measures RoboDojo's two arms) and the brain
    channel (the Rust lane's first-packet text and tool are RoboDojo's)."""
    name = normalize_benchmark(benchmark)
    if name not in REVIEW_BENCHMARKS:
        raise ValueError(f"review mode is implemented for {', '.join(REVIEW_BENCHMARKS)} only, not {name}")
    if name != "robodojo" and isinstance(review, Mapping):
        if "guard" in review:
            raise ValueError(f"review.guard is calibrated on RoboDojo and not ported to {name}")
        if "context" in review:
            raise ValueError(f"review.context (the declared context manifest) is specified on RoboDojo only, not {name}")
        if review.get("schedule", "every_chunk") != "every_chunk" or "on_stall" in review:
            raise ValueError(f"review.schedule on_stall runs on RoboDojo only, not {name}")
        if (review.get("teacher") or {}).get("channel", "python") != "python":
            raise ValueError(f"review.teacher.channel brain runs on RoboDojo only; {name} uses the python channel")
    return name


def _secret_like(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(("sk-", "sk_")) or "Bearer " in value
    if isinstance(value, Mapping):
        return any(_secret_like(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_secret_like(v) for v in value)
    return False


def validate_review(value: Mapping | None) -> dict:
    """Review-mode timing, budget and teacher settings. The teacher's key never appears in a configuration: only the
    path of a private file holding it (read at run time by the teacher)."""
    if not isinstance(value, Mapping):
        raise ValueError("review mode requires a review block")
    review = {**copy.deepcopy(REVIEW_DEFAULTS), **copy.deepcopy(dict(value))}
    unknown = (set(review) - set(REVIEW_DEFAULTS)
               - {"teacher", "task_source", "limit_sources", "schedule", "on_stall", "guard", "context",
                  "llm_token_budget", "task_protocol", "exploration_probe"}
               - set(DESIGN_REVIEW_KEYS) - {"digest_max_chunks"})
    if unknown:
        raise ValueError(f"unknown review settings: {sorted(unknown)}")
    # The review guard (oeh_policy.review_guard), under either schedule. Absent, nothing is added: the review block and
    # its hash stay as they were.
    if "guard" in review:
        from .review_guard import validate_guard
        review["guard"] = validate_guard(review["guard"])
    # OEH hybrid v4 (oeh_policy.takeover): `schedule: on_stall` and its `on_stall` block. Absent, the schedule is
    # every_chunk and the configuration (and its hash) is the review mode's as the 9c5bbac pass froze it.
    from .takeover import SCHEDULES, validate_on_stall
    if "schedule" in review:
        if review["schedule"] not in SCHEDULES:
            raise ValueError(f"review.schedule must be one of {list(SCHEDULES)}")
        # The segment schedule (G13) wakes early on the same detectors, so it carries the on_stall block too.
        if (review["schedule"] in ("on_stall", "segment")) != ("on_stall" in review):
            raise ValueError("review.on_stall is set exactly when review.schedule is on_stall or segment")
        if "on_stall" in review:
            review["on_stall"] = validate_on_stall(review["on_stall"])
    elif "on_stall" in review:
        raise ValueError("review.on_stall needs review.schedule: on_stall")
    validate_design_review_keys(review)
    if "exploration_probe" in review:
        probe = review["exploration_probe"]
        if not isinstance(probe, dict) or set(probe) != {"objective", "diagnosis", "recovery", "check"}:
            raise ValueError("review.exploration_probe requires objective, diagnosis, recovery and check")
        forbidden = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(PACKET_FORBIDDEN_KEYS)) +
                               r")(?![A-Za-z0-9_])", re.IGNORECASE)
        for name, text in probe.items():
            if not isinstance(text, str) or not text.strip() or len(text) > 1000 or forbidden.search(text):
                raise ValueError(f"exploration_probe.{name} must be bounded public text without oracle terms")
        if review.get("schedule") not in (None, "every_chunk") or \
                (review.get("teacher") or {}).get("channel", "python") != "python":
            raise ValueError("exploration_probe requires every-chunk Review on the Python teacher channel")
    if review.get("llm_token_budget") is not None:
        from .llm_budget import validate as validate_llm_budget
        review["llm_token_budget"]=validate_llm_budget(review["llm_token_budget"])
    if review.get("task_protocol") is not None and (review["task_protocol"]!="press_by_number-public-v1" or review.get("task")!="press_by_number"):
        raise ValueError("unknown or mismatched review.task_protocol")
    given = dict(review.get("teacher") or {})
    teacher = {**copy.deepcopy(TEACHER_DEFAULTS), **copy.deepcopy(given)}
    if "base_url" not in given and teacher["api"] in TEACHER_BASE_URLS:
        teacher["base_url"] = TEACHER_BASE_URLS[teacher["api"]]
    unknown = set(teacher) - set(TEACHER_DEFAULTS) - {"model", "reasoning_effort", "api_key_file",
                                                      "codex_accounts_file", "codex_program",
                                                      "claude_accounts_file", "claude_program", "channel",
                                                      "codex_version", "image_downsample_k", "max_input_chars",
                                                      "context_reset_every", "input_context"}
    if unknown:
        raise ValueError(f"unknown teacher settings: {sorted(unknown)}")
    if review.get("llm_token_budget") is not None and (teacher["api"] not in {"codex_app_server","openai_responses"} or teacher["max_attempts"]!=1 or teacher.get("channel","python")!="python"):
        raise ValueError("episode LLM token monitor requires Python Codex/Responses with max_attempts=1")
    if "input_context" in teacher:
        from .input_context import validate
        teacher["input_context"] = validate(teacher["input_context"])
        if teacher["api"] not in {"codex_app_server","openai_responses"} or teacher["max_attempts"] != 1:
            raise ValueError("bounded input requires Codex/Responses and no automatic retries")
        teacher["context_reset_every"] = 0
    if _secret_like(review) or _secret_like(teacher):
        raise ValueError("a configuration must name the key file, never contain a key")
    if teacher["api"] not in TEACHER_APIS:
        raise ValueError(f"review.teacher.api must be one of {sorted(TEACHER_APIS)}")
    if "channel" in teacher:
        if teacher["channel"] not in TEACHER_CHANNELS:
            raise ValueError(f"review.teacher.channel must be one of {list(TEACHER_CHANNELS)}")
        if teacher["channel"] == "brain" and teacher["api"] not in BRAIN_CHANNEL_APIS:
            raise ValueError(f"review.teacher.channel brain runs only the {sorted(BRAIN_CHANNEL_APIS)} teacher so "
                             f"far, not {teacher['api']}")
    if not isinstance(teacher.get("model"), str) or not teacher["model"]:
        raise ValueError("review.teacher.model required")
    # The Codex CLI pin a study lock records (declared-factor-mode.md §1.4): a plain string, kept as given. Absent,
    # the teacher block and its hash are as the formal passes froze them.
    if "codex_version" in teacher and (not isinstance(teacher["codex_version"], str) or not teacher["codex_version"]):
        raise ValueError("review.teacher.codex_version must be a version string (e.g. 0.159.2)")
    if "codex_version" in teacher and teacher["api"] != "codex_app_server":
        raise ValueError("review.teacher.codex_version pins the Codex CLI: it goes with api codex_app_server only")
    efforts = ANTHROPIC_EFFORTS if teacher["api"] in CLAUDE_APIS else EFFORTS
    if teacher.get("reasoning_effort") not in efforts:
        raise ValueError(f"review.teacher.reasoning_effort must be one of {sorted(efforts)}")
    # Each transport names where its credentials live, never the credentials: a key file for an API, a file listing
    # Codex login directories for the app-server (the logins stay in those directories).
    source = TEACHER_CREDENTIALS[teacher["api"]]
    path = teacher.get(source)
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"review.teacher.{source} must be an absolute path")
    for key, default in (("codex_program", "codex"), ("claude_program", "claude")):
        program = teacher.get(key, default)
        if not isinstance(program, str) or not program:
            raise ValueError(f"review.teacher.{key} must name the {default} program")
    if teacher["api"] == "openai_chat" and "base_url" not in given:
        raise ValueError("review.teacher.base_url required for openai_chat (the provider's OpenAI-compatible endpoint)")
    for key in ("image_max_edge", "max_attempts", "max_invalid", "prune_above_tokens", "context_images"):
        if type(teacher[key]) is not int or teacher[key] < 1:
            raise ValueError(f"positive integer review.teacher.{key} required")
    if "context_reset_every" in teacher and (type(teacher["context_reset_every"]) is not int or teacher["context_reset_every"]<0):
        raise ValueError("context_reset_every must be nonnegative; zero preserves the conversation")
    for key in ("image_downsample_k",):
        if key in teacher and (type(teacher[key]) is not int or teacher[key] < 1):
            raise ValueError(f"positive integer review.teacher.{key} required")
    max_chars = teacher.get("max_input_chars", 750000)
    if max_chars is None:
        if (teacher.get("input_context", {}).get("enforce_budget") is not False
                or teacher.get("channel", "python") != "python"):
            raise ValueError("null max_input_chars requires projection-only input_context and Python channel")
    elif type(max_chars) is not int or not 10000 <= max_chars <= 800000:
        raise ValueError("teacher max_input_chars must be an integer leaving transport headroom")
    if (teacher.get("max_input_chars") or teacher.get("context_reset_every")) and teacher.get("channel", "python") != "python":
        raise ValueError("bounded teacher context requires Python channel")
    timeout = teacher["request_timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
        raise ValueError("finite positive review.teacher.request_timeout_s required")
    delays = teacher["retry_delays_s"]
    if (not isinstance(delays, list) or not delays or
            any(isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 <= d < math.inf for d in delays)):
        raise ValueError("review.teacher.retry_delays_s must list non-negative seconds")
    prices = teacher.get("prices_usd_per_million")
    if prices is not None and (not isinstance(prices, Mapping) or
                               not {"input", "cached_input", "output"} <= set(prices) <= {"input", "cached_input",
                                                                                          "output", "cache_write"}):
        raise ValueError("review.teacher.prices_usd_per_million needs input, cached_input and output "
                         "(and optionally cache_write)")
    if prices is None and teacher["api"] in PAID_TEACHER_APIS:
        raise ValueError(f"review.teacher.prices_usd_per_million is required for {teacher['api']}: it is billed per "
                         "token, and the API caps read the usage rows' estimates")
    review["teacher"] = teacher
    # The declared context manifest (declared-factor-mode.md §1). Absent, nothing is added: the review block and
    # its hash stay as they were.
    if "context" in review:
        review["context"] = validate_context(review["context"], review)
    for key in ("call_seconds", "hold_reserve_s", "teacher_max_wait_s", "turn_seconds", "control_hz"):
        if isinstance(review[key], bool) or not isinstance(review[key], (int, float)) or not 0 < review[key] < math.inf:
            raise ValueError(f"finite positive review.{key} required")
    if not review["hold_reserve_s"] < review["call_seconds"] <= review["teacher_max_wait_s"] < review["turn_seconds"]:
        raise ValueError("review timing needs hold_reserve_s < call_seconds <= teacher_max_wait_s < turn_seconds")
    extra = review["first_call_extra_seconds"]
    if isinstance(extra, bool) or not isinstance(extra, (int, float)) or not 0 <= extra < math.inf:
        raise ValueError("non-negative review.first_call_extra_seconds required")
    if type(review["max_decisions"]) is not int or review["max_decisions"] < 0:
        raise ValueError("non-negative integer review.max_decisions required (0: the native step limit only)")
    if type(review["empty_candidate_holds"]) is not int or review["empty_candidate_holds"] < 1:
        raise ValueError("positive integer review.empty_candidate_holds required")
    # A teacher turn spans the wait, its last model call with every retry, and the execution of its rows.
    needed = (review["teacher_max_wait_s"] + teacher["max_attempts"] * timeout
              + sum(delays[min(i, len(delays) - 1)] for i in range(teacher["max_attempts"] - 1)) + 120)
    if review["turn_seconds"] < needed:
        raise ValueError(f"review.turn_seconds {review['turn_seconds']} is below the {needed:g} s a teacher turn may take")
    from .review import validate_limits
    review["limits"] = validate_limits(review.get("limits"))
    return review


def validate_design_review_keys(review: Mapping) -> None:
    """The design mechanisms' review keys: each is accepted only once its realization exists on this commit
    (oeh_policy.mechanisms.MECHANISMS_REALIZED); a value the runtime does not realize is refused by name."""
    from .mechanisms import MECHANISMS_REALIZED
    schedule = review.get("schedule", "every_chunk")
    role = "oeh_hybrid_v4" if schedule == "on_stall" else "oeh_review"
    for key in ("digest", "handback"):
        if key in review and review[key] not in MECHANISMS_REALIZED[key].get(role, frozenset()):
            raise ValueError(f"review.{key} {review[key]!r} is not realized on this commit under schedule {schedule} "
                             "(oeh_policy.mechanisms.MECHANISMS_REALIZED)")
    if ("segment" in review) != (schedule == "segment"):
        raise ValueError("review.segment goes with review.schedule: segment, and that schedule needs it")
    if "segment" in review:
        from .takeover import validate_segment
        validate_segment(review["segment"])
    if "digest_max_chunks" in review:
        # G5: how many unreviewed chunks the first packet after them carries in full (oeh_policy.runtime).
        bound = review["digest_max_chunks"]
        if review.get("digest") != "per_chunk" or type(bound) is not int or not 1 <= bound <= 512:
            raise ValueError("review.digest_max_chunks goes with review.digest per_chunk: an integer in [1, 512]")
    if "notes_store" in review:
        from .mechanisms import _notes_store
        _notes_store(role, review["notes_store"])


def default_context() -> dict:
    """The manifest with every switch at its off value: R0, in the registry's order."""
    return {namespace: {key: copy.deepcopy(off) for key, (off, _) in switches.items()}
            for namespace, switches in CONTEXT_SWITCHES.items()}


def _context_file(path: Any, where: str, benchmark: str) -> str:
    from .teacher import prompt_dir
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError(f"{where} must be a relative path under the teacher prompt directory")
    if not (prompt_dir(benchmark) / path).is_file():
        raise ValueError(f"{where} names no file under the teacher prompt directory: {path}")
    return path


def validate_context(value: Mapping | None, review: Mapping, benchmark: str = "robodojo") -> dict:
    """``review.context``: known namespaces and switches only, every switch left out at its off value, typed and
    bounded values, paired switches together, and no switch off its off value unless the runtime realizes it
    (CONTEXT_REALIZED). The returned manifest holds every namespace and switch in the registry's order, so two
    manifests that mean the same hash the same."""
    if not isinstance(value, Mapping):
        raise ValueError("review.context must be a mapping of namespaces (task, state, behavior, supervisor)")
    unknown = set(value) - set(CONTEXT_SWITCHES)
    if unknown:
        raise ValueError(f"unknown review.context namespaces: {sorted(unknown)}")
    manifest = default_context()
    for namespace, switches in CONTEXT_SWITCHES.items():
        given = value.get(namespace, {})
        if not isinstance(given, Mapping):
            raise ValueError(f"review.context.{namespace} must be a mapping")
        unknown = set(given) - set(switches)
        if unknown:
            raise ValueError(f"unknown review.context.{namespace} settings: {sorted(unknown)}")
        for key, (off, kind) in switches.items():
            if key not in given:
                continue
            where = f"review.context.{namespace}.{key}"
            item = given[key]
            if kind is bool:
                if type(item) is not bool:
                    raise ValueError(f"{where} must be a boolean")
            elif kind == "file":
                if item is not None:
                    item = _context_file(item, where, benchmark)
            elif kind == "history":
                item = _validate_history(item, where)
            elif isinstance(kind, tuple) and kind[0] == "int":
                _, low, high = kind
                if type(item) is not int or not low <= item <= high:
                    raise ValueError(f"{where} must be an integer in [{low}, {high}]")
            elif item not in kind:
                raise ValueError(f"{where} must be one of {list(kind)}")
            manifest[namespace][key] = item
    for namespace, key, on, file_key in CONTEXT_FILE_PAIRS:
        if (manifest[namespace][key] == on) != (manifest[namespace][file_key] is not None):
            raise ValueError(f"review.context.{namespace}.{key}: {on} and {file_key} go together")
    state, behavior, supervisor = manifest["state"], manifest["behavior"], manifest["supervisor"]
    if (state["placebo_tokens"] > 0) != (state["placebo_file"] is not None):
        raise ValueError("review.context.state.placebo_tokens above 0 and placebo_file go together")
    if (behavior["examples_k"] > 0) != (behavior["examples_file"] is not None):
        raise ValueError("review.context.behavior.examples_k above 0 and examples_file go together")
    if state["history"]["policy"] != "transport" and (review.get("teacher") or {}).get("api") != "openai_responses":
        raise ValueError("review.context.state.history.policy other than transport needs review.teacher.api "
                         "openai_responses (the Codex conversation cannot be pruned)")
    if supervisor["channel"] is not None and not (review.get("guard") or {}).get("enforce"):
        raise ValueError("review.context.supervisor.channel needs review.guard with enforce true")
    moved = [f"{namespace}.{key}" for namespace, switches in CONTEXT_SWITCHES.items()
             for key, (off, _) in switches.items() if manifest[namespace][key] != off]
    unrealized = [name for name in moved if name not in CONTEXT_REALIZED]
    if unrealized:
        raise ValueError(f"review.context.{unrealized[0]} is declared on but not realized on this commit; a lock may "
                         "declare a switch only once the runtime realizes it (contracts.CONTEXT_REALIZED)")
    from .review_context import check_recipe, prompt_moved
    if manifest["task"]["note"] == "recipe":
        check_recipe(manifest["task"]["recipe_file"], review.get("task"), benchmark)
    if prompt_moved(manifest) and (review.get("teacher") or {}).get("channel", "python") != "python":
        # The brain channel's lane reads the prompt teacher_bridge.TeacherBridge builds from the schedule alone.
        raise ValueError("review.context switches that change the developer prompt (task.note none, "
                         "state.embodiment, behavior.lessons_file) need review.teacher.channel python on this commit")
    return manifest


def _validate_history(value: Any, where: str) -> dict:
    off = CONTEXT_SWITCHES["state"]["history"][0]
    if not isinstance(value, Mapping) or set(value) - set(off):
        raise ValueError(f"{where} takes exactly the keys {sorted(off)}")
    history = {**off, **dict(value)}
    if history["policy"] not in HISTORY_POLICIES:
        raise ValueError(f"{where}.policy must be one of {list(HISTORY_POLICIES)}")
    for key, (low, high) in HISTORY_BOUNDS.items():
        item = history[key]
        if item is not None and (type(item) is not int or not low <= item <= high):
            raise ValueError(f"{where}.{key} must be an integer in [{low}, {high}] or null")
    anchored = history["policy"] == "anchored"
    if anchored != (history["recent"] is not None and history["max_anchors"] is not None):
        raise ValueError(f"{where}: policy anchored, recent and max_anchors go together")
    return history


def context_identity(manifest: Mapping) -> dict:
    """What identity.json records of a declared manifest: the manifest and its hash."""
    return {"manifest": plain(manifest), "sha256": digest(manifest)}


def packet_field_set(review: Mapping) -> str:
    """The name of the top-level field set a review block's packets carry (declared-factor-mode.md §4.1)."""
    parts = ["v1"]
    if (review.get("guard") or {}).get("enforce"):
        parts.append("+guard")
    if review.get("schedule") in ("on_stall", "periodic", "segment"):
        parts.append("+takeover")
    if review.get("digest") in {"per_chunk", "event_summary"}:
        parts.append("+digest")
    context = review.get("context") or {}
    if (context.get("supervisor") or {}).get("arrival_check"):
        parts.append("+arrival")
    if (context.get("state") or {}).get("episode_record"):
        parts.append("+record")
    if (context.get("state") or {}).get("placebo_tokens"):
        parts.append("+padding")
    if (context.get("task") or {}).get("step_budget") is False:
        parts.append("-step")
    if (context.get("task") or {}).get("decision_budget"):
        parts.append("+decisions")
    if review.get("notes_store"):
        parts.append("+notes")
    if review.get("exploration_probe"):
        parts.append("+probe")
    return "".join(parts)


def packet_fields(field_set: str) -> tuple[set, set]:
    """A field set's (required, allowed) top-level keys."""
    tokens = re.findall(r"[+-][a-z]+", field_set)
    if not field_set.startswith("v1") or "".join(tokens) != field_set[2:]:
        raise ValueError(f"unknown packet field set {field_set!r}")
    required, allowed = set(PACKET_FIELDS_V1), set(PACKET_FIELDS_V1) | set(PACKET_OPTIONAL_V1)
    for token in tokens:
        table = PACKET_ADDITIONS if token[0] == "+" else PACKET_REMOVALS
        if token[1:] not in table:
            raise ValueError(f"unknown packet field set {field_set!r}")
        keys = set(table[token[1:]])
        if token[0] == "+":
            if token[1:] not in PACKET_OPTIONAL_ADDITIONS:
                required |= keys
            allowed |= keys
        else:
            required -= keys
            allowed -= keys
    return required, allowed


def _forbidden_keys(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if str(key) in PACKET_FORBIDDEN_KEYS:
                found.append(where)
            found += _forbidden_keys(item, where)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found += _forbidden_keys(item, f"{path}[{index}]")
    return found


def check_packet(packet: Mapping, field_set: str | None = None) -> None:
    """Refuse a teacher packet that carries a forbidden key at any depth or, with a field set (a declared manifest),
    a top-level key outside the set or a required key missing. Reads only. The oracle boundary is what the harness
    shows the teacher: ``previous_result.assessment`` is the teacher's own previous answer, stored as it came, so a
    key the model chose to write there (``success: "not yet"``) carries no oracle data and is not scanned; every
    harness-built field of ``previous_result`` is."""
    scanned = dict(packet)
    previous = scanned.get("previous_result")
    if isinstance(previous, Mapping) and "assessment" in previous:
        scanned["previous_result"] = {key: value for key, value in previous.items() if key != "assessment"}
    forbidden = _forbidden_keys(scanned)
    if forbidden:
        raise ValueError(f"teacher packet carries forbidden keys (the oracle boundary): {forbidden}")
    if field_set is None:
        return
    required, allowed = packet_fields(field_set)
    extra, missing = sorted(set(packet) - allowed), sorted(required - set(packet))
    if extra or missing:
        raise ValueError(f"teacher packet is outside its declared field set {field_set}: "
                         + "; ".join(filter(None, (f"undeclared {extra}" if extra else "",
                                                   f"missing {missing}" if missing else ""))))
