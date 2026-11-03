"""Compact welcome, persistent input history and slash-command completion."""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .commands import COMMANDS


def render(console, result):
    """Keep the interactive transcript compact; scripted commands retain JSON."""
    if isinstance(result, str):
        console.print(Text(result))
        return
    if (
        isinstance(result, dict)
        and "recipe" in result
        and result.get("scope") == "local software example"
    ):
        console.print("Local example loaded · memory → tool → skill", style="#4dbaac")
        return
    if isinstance(result, dict) and "steps" in result:
        console.print(
            Text(
                result["name"]
                + " · "
                + " → ".join(step["name"] for step in result["steps"])
            )
        )
        return
    if (
        isinstance(result, dict)
        and "output" in result
        and result.get("state") == "completed"
    ):
        console.print(Text("Completed · " + result["id"][:8], style="#4dbaac"))
        console.print_json(json.dumps(result["output"]))
        console.print(
            "Software execution only · no physical qualification", style="dim"
        )
        return
    if isinstance(result, dict) and result.get("state") in {
        "inherit_child",
        "retained",
    }:
        console.print(
            Text(result["state"] + " · " + result["harness"], style="#4dbaac")
        )
        console.print(
            Text(result.get("reason", "Local Self-Harness comparison completed"))
        )
        console.print(Text(result["scope"], style="dim"))
        return
    if isinstance(result, dict) and all(key.startswith("/") for key in result):
        for name, description in result.items():
            text = Text(f"{name:12}", style="#4dbaac")
            text.append(description, style="default")
            console.print(text)
        return
    console.print_json(json.dumps(result, ensure_ascii=False))


def banner(console, workspace, *, plain=False):
    title = Text("physical", style="bold")
    title.append("RSI", style="bold #4dbaac")
    if (
        not plain
        and console.width >= 76
        and console.encoding.lower().replace("-", "") == "utf8"
    ):
        # Glyphs sampled from the website's actual Grenze Gotisch 500 font.
        logo = Path(__file__).with_name("assets").joinpath("wordmark.txt").read_text()
        console.print(Text(logo.rstrip(), style="#4dbaac"))
    body = Text("Composable embodied intelligence\n", style="dim")
    body.append(str(workspace) + "\n\n", style="dim")
    body.append("/demo  /task  /scheme  /harness  /baseline  /help", style="#4dbaac")
    body.append("\n/model config.json · configure conversation", style="dim")
    console.print(Panel(body, title=title, title_align="left", border_style="#4dbaac"))


def run_console(application, commands=None, *, plain=False):
    console = Console(highlight=False, no_color=plain)
    if commands is None:
        banner(console, application.workspace, plain=plain)

    def run(line):
        try:
            result = application.command(line)
            if result is not None:
                if commands is None:
                    render(console, result)
                else:
                    console.print_json(json.dumps(result, ensure_ascii=False))
            return 0
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            console.print(Text(str(error), style="red"))
            return 2

    if commands is not None:
        for line in commands:
            try:
                code = run(line)
            except EOFError:
                return 0
            if code:
                return code
        return 0

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    application.workspace.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(application.workspace / "history")),
        completer=WordCompleter(["/" + name for name in COMMANDS], sentence=True),
        style=Style.from_dict({"prompt": "#4dbaac bold"}),
    )
    while True:
        try:
            line = session.prompt([("class:prompt", "❯ " if not plain else "> ")])
            run(line)
        except EOFError:
            return 0
        except KeyboardInterrupt:
            continue
