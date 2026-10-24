import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def table(rows, columns):
    """Rows of dicts as an aligned text table."""
    cells = [[("" if row.get(column) is None else str(row.get(column))) for column in columns] for row in rows]
    widths = [max([len(column)] + [len(line[i]) for line in cells]) for i, column in enumerate(columns)]
    lines = ["  ".join(column.ljust(width) for column, width in zip(columns, widths))]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)) for line in cells]
    return "\n".join(line.rstrip() for line in lines)


def _deploy_argument(command):
    command.add_argument("--deploy", help="deploy name (configs/deploy/<name>.toml) or path; default $HARNESS_DEPLOY")


def _model_arguments(command):
    command.add_argument("--model", help="model roles: a name of configs/models/ (e.g. grok-low) or a .toml file")
    command.add_argument("--agent", type=Path, help="agent file in place of configs/default.toml "
                                                    "(e.g. configs/ablations/r2.toml)")


def _run_arguments(command):
    _deploy_argument(command)
    _model_arguments(command)
    command.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                         help="override an experiment key, e.g. budget.max_cost_usd=2, agent.runtime.keep_images=1, "
                              "model.main.reasoning_effort=low (repeatable)")
    command.add_argument("--tasks", help="tasks overriding the experiment's: names, dimensions, moving, all")
    command.add_argument("--seeds", help="seeds overriding the experiment's, e.g. 0-4 or 0,2,5")
    command.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None,
                         help="refuse to run without a skill or service the agent names "
                              "(default: on for evaluation layouts)")


def parser():
    root = argparse.ArgumentParser(prog="harness", description="The Harness: models write programs that call skills")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the agent over the tasks and seeds of an experiment "
                                          "(configs/experiments/*.toml, or --tasks and --seeds)")
    run.add_argument("config", type=Path, nargs="?", help="experiment file (default: none; give --tasks)")
    _run_arguments(run)
    run.add_argument("--output", type=Path, required=True, help="suite directory (resumes when it exists)")
    run.add_argument("--gpus", help="GPUs for simulator workers, e.g. 3,5 (default: the deploy's [sim] gpus)")
    run.add_argument("--workers-per-gpu", type=int)
    run.add_argument("--base-port", type=int)
    run.add_argument("--rerun-errors", action="store_true", help="run again the episodes that ended in an error")
    run.add_argument("--max-total-cost-usd", type=float, help="start no new episode once the suite spent this much")
    run.add_argument("--script", type=Path, help="JSON replies to replay instead of calling a model")
    program = commands.add_parser("program", help="Run the seeds of one task against a worker that already runs")
    program.add_argument("config", type=Path, nargs="?", help="experiment file (configs/experiments/*.toml)")
    _run_arguments(program)
    program.add_argument("--task", help="task (default: the experiment's first)")
    program.add_argument("--endpoint", help="worker URL (default: the experiment's endpoint)")
    program.add_argument("--output", type=Path, default=Path("runs"))
    program.add_argument("--script", type=Path, help="JSON list of model replies to replay instead of calling a model")
    config = commands.add_parser("config", help="Show what the agent (or an experiment) resolves to on a deploy: "
                                                "skills, services, identity")
    config.add_argument("config", type=Path, nargs="?",
                        help="experiment file or agent file (default: configs/default.toml)")
    _run_arguments(config)
    config.add_argument("--json", action="store_true", help="print the whole resolved configuration")
    lock = commands.add_parser("lock", help="Write the agent's lock: what decides its behavior, and its identity")
    lock.add_argument("config", type=Path, nargs="?", help="agent file or experiment file (default: configs/default.toml)")
    _run_arguments(lock)
    lock.add_argument("--out", type=Path, help="write the lock here (default: print it)")
    export = commands.add_parser("export", help="Export the agent as a policy another evaluator runs")
    targets = export.add_subparsers(dest="target", required=True)
    xpl = targets.add_parser("xpolicylab", help="An XPolicyLab policy folder (policy/<Name>/) for RoboDojo evaluation")
    xpl.add_argument("config", type=Path, nargs="?", help="agent file or experiment file (default: configs/default.toml); "
                                                          "its runtime.contract must be xpolicylab")
    xpl.add_argument("--name", required=True, help="the policy's folder and module name, e.g. HarnessPolicy")
    xpl.add_argument("--out", type=Path, required=True, help="the folder to write (XPolicyLab/policy/<Name>)")
    _deploy_argument(xpl)
    xpl.add_argument("--model", help="model roles: a name of configs/models/ or a .toml file")
    xpl.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    xpl.add_argument("--vendor", action="store_true", help="copy this repository into the folder (the-harness/) "
                                                           "instead of letting install.sh clone it")
    xpl.add_argument("--xpolicylab", type=Path, help="XPolicyLab checkout (default: the deploy's)")
    xpl.add_argument("--allow-dirty", action="store_true", help="export with uncommitted code changes")
    xpl.add_argument("--embodiment", default="arx_x5", help="the robot: a name of embodiments/ (default arx_x5, "
                                                          "RoboDojo's); it must be calibrated")
    xpl.add_argument("--check", action="store_true", help="then run XPolicyLab's PR checks and the debug closed loop")
    xpl.add_argument("--gpu", default="0", help="--check: policy GPU ids, e.g. 2,3")
    xpl.add_argument("--env", default="harness-policy", help="--check: conda environment of policy and client")
    xpl.add_argument("--script", type=Path, help="--check: model replies to replay "
                                                 "(default: configs/scripts/xpolicylab_check.json)")
    report = commands.add_parser("report", help="RoboDojo Score/SR table over program-harness runs")
    report.add_argument("runs", type=Path, nargs="+", help="experiment directories (searched recursively)")
    report.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    services = commands.add_parser("services", help="List the services (services/*.toml)")
    services.add_argument("--json", action="store_true")
    for name, text in (("up", "Start the resident services of a deploy (all, or the ones named)"),
                       ("down", "Stop services that harness up started"),
                       ("status", "Where each service of a deploy runs and whether it answers"),
                       ("doctor", "Check that this machine can run a deploy: files, GPUs, ports, interpreters")):
        command = commands.add_parser(name, help=text)
        if name in ("up", "down", "doctor", "status"):
            command.add_argument("names", nargs="*", help="services (default: all the deploy runs)")
        _deploy_argument(command)
        command.add_argument("--json", action="store_true")
    fetch = commands.add_parser("fetch", help="Download the weights of services into the deploy's weights root")
    fetch.add_argument("names", nargs="+")
    _deploy_argument(fetch)
    serve = commands.add_parser("serve", help="Serve a worker (a robot or simulator) in its own Python process")
    serve.add_argument("--factory", required=True, help="module:callable, e.g. benches.robodojo.runtime:create_environment")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--options", type=Path, help="TOML keyword arguments passed to the factory")
    serve.add_argument("--contract", choices=["native", "xpolicylab"], default="native",
                       help="what clients may see (core/contract.py); xpolicylab: what RoboDojo's official "
                            "client hands a policy")
    serve.add_argument("--embodiment", help="embodiment name (embodiments/<name>/embodiment.toml) for the contract")
    return root


def _services(args):
    from .services import supervisor, tools
    from .services.deploy import load_deploy
    from .services.registry import load_services

    if args.command == "services":
        rows = [{"service": spec.name, "kind": spec.kind, "protocol": spec.protocol, "port": spec.port,
                 "vram_gb": spec.vram_gb, "summary": spec.summary} for spec in load_services().values()]
        output(rows) if args.json else print(table(rows, ["service", "kind", "protocol", "port", "vram_gb", "summary"]))
        return 0
    deploy = load_deploy(args.deploy)
    log = (lambda message: print(message, file=sys.stderr, flush=True))
    if args.command == "fetch":
        return 1 if sum(tools.fetch(deploy, name, log=log) for name in args.names) else 0
    if args.command == "down":
        supervisor.down(deploy, args.names or None, log=log)
        return 0
    if args.command == "doctor":
        report = tools.doctor(deploy, names=args.names or None)
        if args.json:
            output(report)
        else:
            print(f"deploy {report['deploy']}: {'ready' if report['ok'] else 'NOT ready'}")
            if report["gpus"] is not None:
                print(table(report["gpus"], ["index", "name", "used_gb", "total_gb"]))
            for problem in report["problems"]:
                print(f"  problem: {problem}")
            for note in report["notes"]:
                print(f"  note: {note}")
        return 0 if report["ok"] else 1
    rows = supervisor.up(deploy, args.names or None, log=log) if args.command == "up" else \
        supervisor.status(deploy, args.names or None)
    if args.json:
        output(rows)
    else:
        print(table(rows, ["service", "kind", "mode", "gpu", "url", "up", "pid"]))
    return 0 if all(row.get("up") is not False for row in rows) else 1


def _load_experiment(args):
    from .config.experiments import load_experiment
    from .runtime.suite import expand_seeds

    sets = list(args.sets)
    if args.agent:
        sets.append((["agent"], str(args.agent.resolve())))
    return load_experiment(args.config, deploy=args.deploy, sets=sets,
                           tasks=args.tasks.split(",") if getattr(args, "tasks", None) else None,
                           seeds=expand_seeds(args.seeds) if getattr(args, "seeds", None) else None,
                           strict=getattr(args, "strict", None), model=args.model)


def _is_agent_file(args) -> bool:
    """config and lock: no experiment file, or an agent file rather than an experiment."""
    from .config import load_layered
    from .config.experiments import is_experiment

    return args.config is None or not is_experiment(load_layered(args.config))


def _runs(args):
    script = json.loads(args.script.read_text()) if getattr(args, "script", None) else None
    if args.command == "program" and args.task:
        args.tasks = args.task
    if args.command in ("config", "lock") and _is_agent_file(args) and not args.tasks:
        from .config import apply_sets, load_layered, parse_set
        from .config.agent import DEFAULT_AGENT, resolve_agent

        path = args.agent or args.config or DEFAULT_AGENT
        tree = apply_sets(load_layered(path), [parse_set(item) for item in args.sets])
        agent = resolve_agent(tree, model=args.model, deploy=args.deploy, strict=bool(args.strict),
                              name=Path(path).stem)
        run = None
    else:
        if args.command in ("config", "lock") and args.config is not None and _is_agent_file(args):
            args.agent, args.config = args.config, None
        run = _load_experiment(args)
        agent = run.agent
    if args.command == "lock":
        from .config.agent import code_fingerprint

        document = {**agent.lock, "identity": agent.identity, "code": code_fingerprint()}
        text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        if args.out:
            args.out.write_text(text)
            print(json.dumps({"lock": str(args.out), "identity": agent.identity}))
        else:
            print(text, end="")
        return 0
    if args.command == "config":
        from .runtime.skills import load_skills

        if args.json:
            output({"agent": agent.name, "agent_version": agent.lock["agent_version"], "identity": agent.identity,
                    "main": agent.main, "runtime": agent.runtime, "placement": agent.placement, "lock": agent.lock,
                    **({"tasks": run.tasks, "seeds": run.seeds, "layouts": run.layouts, "strict": run.strict,
                        "deploy": run.deploy.name} if run else {})})
            return 0
        print(f"agent {agent.name} {agent.lock['agent_version']}  identity {agent.identity}  mode {agent.lock['mode']}")
        print(f"main model: {agent.main.get('model')} ({agent.main.get('backend', 'openrouter')})")
        if run:
            print(f"bench {run.bench}, {len(run.tasks)} tasks x {len(run.seeds)} seeds, {run.layouts} layouts, "
                  f"deploy {run.deploy.name}, strict {run.strict}")
        dropped = agent.runtime.get("skills_dropped", {})
        rows = [{"skill": name, "on": "yes" if name in agent.runtime["skills"] else "no",
                 "why": dropped.get(name, "" if name in agent.runtime["skills"] else "not in this configuration")}
                for name in load_skills()]
        print(table(rows, ["skill", "on", "why"]))
        rows = []
        for key, value in agent.placement.items():
            if isinstance(value, dict) and "service" in value:
                rows.append({"uses": key, "service": value["service"], "status": value.get("dropped", "")})
            elif isinstance(value, dict):
                rows += [{"uses": key, "service": name, "status": entry.get("dropped", "")}
                         for name, entry in value.items()]
        if rows:
            print(table(rows, ["uses", "service", "status"]))
        return 0
    if args.command == "program":
        from .runtime.driver import run_task

        from .runtime.suite import preflight

        preflight(run, log=lambda message: print(message, file=sys.stderr, flush=True))
        config = run.task_config(args.task, endpoint=args.endpoint)
        if not config.environment.get("endpoint"):
            raise ValueError("No worker endpoint: give --endpoint or set endpoint in the experiment file "
                             "(or use harness run, which starts workers)")
        result = run_task(config, args.output, script=script)
        output(result)
        return 1 if any(item["status"] == "error" for item in result["episodes"]) else 0
    from .runtime.suite import run_suite

    result = run_suite(run, args.output, gpus=[int(g) for g in args.gpus.split(",")] if args.gpus else None,
                       workers_per_gpu=args.workers_per_gpu, base_port=args.base_port,
                       rerun_errors=args.rerun_errors, script=script, max_total_cost_usd=args.max_total_cost_usd)
    output(result)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("run", "program", "config", "lock"):
            return _runs(args)
        if args.command == "report":
            from .runtime.report import aggregate, load_episodes, markdown
            result = aggregate(load_episodes(args.runs))
            if args.json:
                output(result)
            else:
                print(markdown(result))
            return 0
        if args.command in ("services", "up", "down", "status", "doctor", "fetch"):
            return _services(args)
        if args.command == "export":
            from .export.xpolicylab import check, export

            result = export(args.config, name=args.name, out=args.out, deploy=args.deploy, model=args.model,
                            sets=args.sets, vendor=args.vendor, xpolicylab=args.xpolicylab,
                            allow_dirty=args.allow_dirty, embodiment=args.embodiment)
            if args.check:
                result["check"] = check(result["folder"], gpu=args.gpu, env=args.env, script=args.script,
                                        log=lambda message: print(message, file=sys.stderr, flush=True))
            output(result)
            return 0 if result.get("check", {}).get("ok", True) else 1
        if args.command == "serve":
            from .config import read_toml
            from .core.worker import create_server
            module, function = args.factory.split(":", 1)
            factory = getattr(importlib.import_module(module), function)
            contract = None
            if args.contract != "native":
                from .core.contract import ContractFilter
                from .embodiments import load_embodiment

                if not args.embodiment:
                    raise ValueError("--contract needs --embodiment")
                contract = ContractFilter(args.contract, load_embodiment(args.embodiment))
            environment = factory(**(read_toml(args.options) if args.options else {}))
            try:
                with create_server(environment, args.host, args.port,
                                   os.environ.get("HARNESS_WORKER_TOKEN"), contract=contract) as server:
                    print(json.dumps({"worker": environment.name, "address": server.server_address}), flush=True)
                    server.serve_forever()
            finally:
                environment.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, TimeoutError) as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}), file=sys.stderr, flush=True)
        return 1
