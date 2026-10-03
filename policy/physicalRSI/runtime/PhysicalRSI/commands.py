"""One command registry for dispatch, help and completion."""

import json
import shlex

COMMANDS = {
    "help": "Show commands",
    "demo": "Load a local Memory → Tool → Skill example",
    "skill-memory": "Show core policy skills, exploration skill and task memory skill_choices",
    "skills": "List registered skills",
    "tools": "List registered tools",
    "memory": "List pinned memory sources",
    "compose": "<name> <capability>...  Create a serial composition",
    "save": "<path>  Save the exact recipe",
    "load": "<path>  Load a recipe against registered versions",
    "run": "[JSON input]  Execute the current composition",
    "evolve": "Run the local Self-Harness example",
    "status": "Show recipe, workspace and resources",
    "exit": "Exit",
    "model": "[config.json] Configure conversation API, or show configuration",
    "rsi-plan": "[show | set JSON] Select hybrid, learned_skill or code_policy and persist the plan",
    "scheme": "<hybrid | learned_skill | code_policy>  Select an RSI search scheme",
    "harness": "[show | design JSON | import path.json] Configure a data-only custom harness",
    "baseline": "[JSON input] Preflight and run the active task once",
    "chat": "<message> Talk to the configured model",
    "reset-chat": "Clear the current model conversation",
    "task": "<package.json> Load an installed environment/RSI task package",
    "draft": "[show | start <package.json> | set <JSON object> | apply] Prepare task configuration; set recursively updates existing fields, apply loads without running",
    "check": "Check the active task's environment, models and experiment inputs",
    "rsi": "Run one Self-Harness iteration for the active task",
}


def dispatch(application, line):
    text = line.strip()
    if not text:
        return None
    if not text.startswith("/"):
        return application.chat(text)
    command, _, tail = text[1:].partition(" ")
    if command not in COMMANDS:
        raise ValueError(f"Unknown command /{command}; use /help")
    if command in {"help", "demo", "skill-memory", "skills", "tools", "memory", "evolve", "status", "exit"} and tail.strip():
        raise ValueError(f"/{command} takes no arguments")
    if command == "exit":
        raise EOFError
    if command == "draft":
        action, _, value = tail.strip().partition(" ")
        action = action or "show"
        if action == "set":
            return application.task_draft(action, json.loads(value))
        args = shlex.split(value)
        if action == "start" and len(args) == 1:
            return application.task_draft(action, args[0])
        if action in {"show", "apply"} and not args:
            return application.task_draft(action)
        raise ValueError("Use /draft show, start path.json, set JSON, or apply")
    if command == "task":
        args = shlex.split(tail)
        if len(args) != 1:
            raise ValueError("/task requires one task package path")
        return application.load_task(args[0])
    if command == "rsi-plan":
        action, _, value = tail.strip().partition(" ")
        action = action or "show"
        if action == "show" and not value.strip():
            return application.rsi_plan("show")
        if action == "set" and value.strip():
            return application.rsi_plan("set", json.loads(value))
        raise ValueError("Use /rsi-plan show or /rsi-plan set JSON")
    if command == "scheme":
        args = shlex.split(tail)
        if len(args) != 1:
            raise ValueError("/scheme requires hybrid, learned_skill or code_policy")
        return application.rsi_plan("set", {"scheme": args[0]})
    if command == "harness":
        action, _, value = tail.strip().partition(" ")
        action = action or "show"
        if action == "show" and not value.strip():
            return application.harness("show")
        if action == "design" and value.strip():
            return application.harness("design", json.loads(value))
        if action == "import" and value.strip():
            args = shlex.split(value)
            if len(args) == 1:
                return application.harness("import", args[0])
        raise ValueError("Use /harness show, design JSON, or import path.json")
    if command == "baseline":
        if tail.strip():
            value = json.loads(tail)
        else:
            value = None
        return application.run_baseline(value)
    if command in {"check", "rsi"}:
        if tail.strip():
            raise ValueError(f"/{command} takes no arguments")
        return (
            application.check_task()
            if command == "check"
            else application.evolve()
        )
    if command == "help":
        return {"/" + key: value for key, value in COMMANDS.items()}
    if command == "chat":
        if not tail.strip():
            raise ValueError("/chat requires a message")
        return application.chat(tail)
    if command == "model":
        args = shlex.split(tail)
        if len(args) > 1:
            raise ValueError("/model accepts one configuration path")
        return application.configure_model(args[0] if args else None)
    if command == "reset-chat":
        if tail.strip():
            raise ValueError("/reset-chat takes no arguments")
        application.conversation = None
        return {"conversation": "cleared"}
    if command == "demo":
        return application.demo()
    if command == "skill-memory":
        return application.skill_memory_demo()
    if command == "status":
        return application.status()
    if command == "evolve":
        return application.evolve()
    if command in {"memory", "tools", "skills"}:
        prefix = {"memory": "memory.", "tools": "tool.", "skills": "skill."}[command]
        return [
            {"name": op.name, "revision": op.revision}
            for op in application.registry.list()
            if op.name.startswith(prefix)
        ]
    if command == "run":
        return application.run(json.loads(tail) if tail.strip() else None)
    args = shlex.split(tail)
    if command in {"save", "load"}:
        if len(args) != 1:
            raise ValueError(f"/{command} requires one path")
        return getattr(application, command)(args[0])
    if len(args) < 2:
        raise ValueError("/compose requires a name and at least one capability")
    return application.compose(args[0], args[1:])
