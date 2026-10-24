"""Experiments: the agent with a model on a benchmark, as configs/experiments/<name>.toml
describes it.

    agent = "../ablations/r2.toml"          optional: the agent file (default configs/default.toml)
    model = "grok-low"                      optional: configs/models/<name>.toml (default the agent's)
    bench = "robodojo"                      robodojo | robotwin
    tasks = ["general_pickup"]              names, dimension names, "moving", "all"
    seeds = [0]
    layouts = "procedural"                  procedural (development) | eval
    layout_pack_seed = 0                    eval: which official pack
    deploy = "mine"                         optional; else --deploy or $HARNESS_DEPLOY
    strict = true                           default: true for eval layouts
    endpoint = "http://127.0.0.1:8771"      `harness program`: a worker that already runs

    [budget]                                per episode, and for the whole suite
    max_wall_seconds = 5400
    max_cost_usd = 4.0
    max_total_cost_usd = 20.0

    [worker]                                options of the benchmark's worker
    record_video = false

    [overrides.agent]                       changes to the agent for this experiment
    runtime.keep_images = 1
    [overrides.model]                       changes to the model roles
    main.reasoning_effort = "low"

`--set key.path=value` writes into this tree; `--set agent.a.b=v` and
`--set model.a.b=v` go to the overrides, `--set model=NAME` (or `--model NAME`)
picks the model file.
"""

from dataclasses import dataclass, field
from pathlib import Path

from . import apply_sets, deep_merge, load_layered, parse_set
from .agent import DEFAULT_AGENT, ResolvedAgent, load_model, resolve_agent
from ..services.deploy import load_deploy


BENCHES = ("robodojo", "robotwin")
# The embodiment kind each bench's worker drives (embodiments.KINDS).
BENCH_EMBODIMENT = {"robodojo": "robot", "robotwin": "robot"}
LAYOUTS = ("procedural", "eval")
EXPERIMENT_KEYS = {
    "description": "one line",
    "agent": "the agent file (default configs/default.toml; an ablation of configs/ablations/)",
    "model": "the model roles: a name of configs/models/ (default the agent's)",
    "bench": f"one of {', '.join(BENCHES)} (default robodojo)",
    "tasks": "task names, dimension names, \"moving\" or \"all\"",
    "seeds": "a list, or a range such as \"0-49\" (default [0])",
    "layouts": f"one of {', '.join(LAYOUTS)}: development or official evaluation layouts",
    "layout_pack_seed": "eval layouts: which official pack (0, 1, 2)",
    "dev_layout_dir": "development layouts' directory (default <RoboDojo>/harness_dev_layouts/arx_x5)",
    "deploy": "the deploy to resolve against (else --deploy or $HARNESS_DEPLOY)",
    "strict": "fail instead of dropping a skill or service (default: true for eval layouts)",
    "endpoint": "`harness program`: a worker that already runs",
    "timeout_seconds": "worker request timeout (default 600)",
    "budget": "max_wall_seconds, max_cost_usd, max_model_calls per episode; max_total_cost_usd per suite",
    "worker": "options of the bench's worker (record_video, ground_truth, task_config, seed_cache_dir)",
    "suite": "options of `harness run` (boot_timeout, ...)",
    "overrides": "[overrides.agent] and [overrides.model]: changes for this experiment only",
}
# Keys that make a file an experiment rather than an agent file.
EXPERIMENT_ONLY = ("agent", "bench", "tasks", "seeds", "layouts", "layout_pack_seed", "endpoint")


def is_experiment(tree: dict) -> bool:
    """Whether a loaded file is an experiment (it names a bench, tasks, seeds, layouts or
    an agent) rather than an agent file."""
    return bool(set(tree) & set(EXPERIMENT_ONLY))


@dataclass
class Experiment:
    """A loaded experiment: its tree (with --set items applied), the agent resolved
    against the deploy, the expanded tasks and seeds, and the experiment's budget,
    worker and suite options. `task_config` gives the driver's configuration for one
    task."""

    path: Path | None
    tree: dict
    agent: ResolvedAgent
    deploy: object
    bench: str
    tasks: list[str]
    seeds: list[int]
    layouts: str
    layout_pack_seed: int
    strict: bool
    budget: dict = field(default_factory=dict)
    worker: dict = field(default_factory=dict)
    suite: dict = field(default_factory=dict)
    endpoint: str | None = None
    timeout_seconds: float = 600.0

    def task_config(self, task: str | None = None, seeds=None, endpoint: str | None = None):
        """The driver's configuration for one task (default: the experiment's first)."""
        from ..runtime.driver import TaskConfig

        task = task or self.tasks[0]
        seeds = list(seeds if seeds is not None else self.seeds)
        if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a nonempty list of distinct integers")
        endpoint = endpoint or self.endpoint
        agent = self.agent
        cost = self.budget.get("max_cost_usd")
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0):
            raise ValueError("budget.max_cost_usd must be a positive number")
        return TaskConfig(
            path=self.path or Path("."), environment={"endpoint": endpoint, "timeout_seconds": self.timeout_seconds},
            main=dict(agent.main), runtime=dict(agent.runtime), system_prompt=agent.system_prompt, task=task,
            seeds=seeds, max_model_calls=int(self.budget.get("max_model_calls", agent.max_model_calls)),
            max_wall_seconds=float(self.budget.get("max_wall_seconds", 3600)), verifier=dict(agent.verifier),
            base_prompt=agent.base_prompt, task_description=task_description(agent.runtime, task),
            max_cost_usd=None if cost is None else float(cost), lock=agent.lock, agent_id=agent.identity,
            placement={"deploy": getattr(self.deploy, "name", None), **agent.placement},
            embodiment=agent.embodiment)


def task_description(runtime: dict, task: str) -> str:
    """The task's description (RoboDojo wiki recipe) when the agent uses them."""
    directory = runtime.get("task_descriptions")
    if not directory:
        return ""
    source = Path(directory) / f"{task.removesuffix('_random')}.md"
    if not source.is_file():
        raise ValueError(f"No task description for {task} in {directory}")
    return source.read_text(encoding="utf-8").strip()


def _sets(sets):
    """--set items, with agent.* and model.* moved under overrides (model=NAME stays: it
    picks the model file)."""
    out = []
    for item in sets or ():
        keys, value = parse_set(item) if isinstance(item, str) else item
        if keys and keys[0] in ("agent", "model") and len(keys) > 1:
            keys = ["overrides", *keys]
        out.append((keys, value))
    return out


def load_experiment(path=None, *, tree: dict | None = None, deploy=None, sets=(), tasks=None, seeds=None,
                    strict: bool | None = None, model=None) -> Experiment:
    """An experiment file (or tree; neither: an empty experiment, the default agent on
    RoboDojo's development layouts) with --set items applied, its agent resolved against
    the deploy (argument, the file's `deploy`, or $HARNESS_DEPLOY). `model` (a model
    name or file) replaces the experiment's."""
    if tree is None:
        if path is not None:
            path = Path(path).resolve()
            tree = load_layered(path)
        else:
            tree = {}
    tree = apply_sets(tree, _sets(sets))
    if model is not None:
        tree["model"] = str(model)
    unknown = set(tree) - set(EXPERIMENT_KEYS)
    if unknown:
        raise ValueError(f"Experiment {path}: unknown keys {sorted(unknown)}")
    bench = tree.get("bench", "robodojo")
    if bench not in BENCHES:
        raise ValueError(f"bench must be one of {BENCHES}")
    layouts = tree.get("layouts", "procedural")
    if layouts not in LAYOUTS:
        raise ValueError(f"layouts must be one of {LAYOUTS}")
    if tasks is not None:
        tree["tasks"] = tasks
    if seeds is not None:
        tree["seeds"] = seeds
    if not tree.get("tasks"):
        raise ValueError(f"Experiment {path or '(none)'}: names no tasks (give --tasks)")
    from ..runtime.suite import expand_seeds, expand_tasks

    task_list = tree["tasks"]
    task_list = expand_tasks(task_list if isinstance(task_list, str) else ",".join(task_list), bench)
    seed_list = tree.get("seeds", [0])
    seed_list = expand_seeds(seed_list) if isinstance(seed_list, str) else [int(seed) for seed in seed_list]
    strict = (layouts == "eval") if strict is None and tree.get("strict") is None else (
        strict if strict is not None else bool(tree["strict"]))
    deploy = deploy if deploy is not None and not isinstance(deploy, (str, Path)) else load_deploy(
        deploy or tree.get("deploy"))
    overrides = tree.get("overrides", {})
    unknown = set(overrides) - {"agent", "model"}
    if unknown:
        raise ValueError(f"Experiment {path}: overrides are [overrides.agent] and [overrides.model], "
                         f"not {sorted(unknown)}")
    agent_path = Path(tree.get("agent") or DEFAULT_AGENT)
    agent_tree = deep_merge(load_layered(agent_path), overrides.get("agent", {}))
    model_tree = deep_merge(load_model(tree.get("model") or agent_tree.get("model")), overrides.get("model", {}))
    agent = resolve_agent(agent_tree, model=model_tree, deploy=deploy, strict=strict, name=agent_path.stem,
                          runtime_overrides=overrides.get("agent", {}).get("runtime"),
                          embodiment=BENCH_EMBODIMENT[bench])
    return Experiment(path=path, tree=tree, agent=agent, deploy=deploy, bench=bench, tasks=task_list,
                      seeds=seed_list, layouts=layouts, layout_pack_seed=int(tree.get("layout_pack_seed", 0)),
                      strict=strict, budget=dict(tree.get("budget", {})), worker=dict(tree.get("worker", {})),
                      suite=dict(tree.get("suite", {})), endpoint=tree.get("endpoint"),
                      timeout_seconds=float(tree.get("timeout_seconds", 600)))
