#!/usr/bin/env python3
"""Write static HTML views of a runs/ directory for reading results by eye.

    python3 scripts/view_runs.py runs            # then open runs/index.html
    bash scripts/run_and_view.sh EXPERIMENT ...   # harness program, then this

runs/index.html          one row per experiment, aggregated; dev and eval in separate tables
<experiment>/index.html  the experiment's episodes
<episode>/view.html      the episode's recorded events in order: prompts and images,
                         replies, executed actions and their results, the outcome

Reads summary.json, manifest.json and events.jsonl as written and changes none of them;
the pages are written next to them, inside the (git-ignored) runs directory. Pages are
static: rerun after new or deleted runs. Standard library only.
Scores are `harness report`'s (src/harness_policy/runtime/report.py): an episode counts 1 on
success, otherwise process_score / 100, an error counts 0, and an episode the simulator dropped as
unstable is not scored. Each suite folder also gets the report's Score / SR by dimension.
"""

import ast
from html import escape
import io
import json
import keyword
from pathlib import Path
import re
import sys
import time
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.protocol import ProtocolError, parse_reply  # noqa: E402
from harness_policy.runtime.report import DIMENSIONS, aggregate, episode_score  # noqa: E402

STYLE = """
body { font: 14px/1.45 system-ui, sans-serif; margin: 20px; color: #222; background: #fafafa; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
table { border-collapse: collapse; background: #fff; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; } td.n { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #137333; font-weight: 600; } .bad { color: #b3261e; font-weight: 600; }
.tag { font-size: 12px; padding: 1px 6px; border-radius: 4px; background: #e8eaed; }
.eval { background: #fde7c7; }
.event { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px; margin: 10px 0; }
.event h3 { font-size: 13px; margin: 0 0 6px; color: #555; }
.event { border-left: 5px solid #bbb; }
.e-input { border-left-color: #c4c7c5; } .e-reply { border-left-color: #1a73e8; }
pre.code { counter-reset: line; background: #f8f9fa; border: 1px solid #e3e3e3; }
pre.code .l { display: block; padding-left: 3.4em; text-indent: -3.4em; }
pre.code .l::before { counter-increment: line; content: counter(line); display: inline-block; width: 2.6em;
  margin-right: 0.8em; text-align: right; color: #9aa0a6; text-indent: 0; user-select: none; }
pre.code .k { color: #8430ce; } pre.code .s { color: #188038; } pre.code .n { color: #b06000; }
pre.code .c { color: #6b6b6b; font-style: italic; } pre.code .a { color: #1a57c8; font-weight: 700; }
pre.code .x { cursor: pointer; border-bottom: 1px dotted #188038; }
pre.prose { background: #fff; border-left: 3px solid #c4c7c5; }
.outline { font-size: 13px; margin: 6px 0; } .outline ol { margin: 2px 0 2px 18px; padding: 0; }
.e-ok { border-left-color: #1e8e3e; } .e-bad { border-left-color: #d93025; } .e-esc { border-left-color: #e37400; }
.esc { color: #8a4b00; font-weight: 600; }
.e-frame { border-left-color: #5f6368; } .e-verdict { border-left-width: 9px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0; background: #f6f6f6; padding: 6px; }
figure { display: inline-block; margin: 2px 6px 2px 0; } figcaption { font-size: 12px; color: #666; }
img { height: 150px; border: 1px solid #ccc; cursor: zoom-in; }
img:focus { height: 480px; max-width: 100%; object-fit: contain; outline: 2px solid #1a73e8; cursor: zoom-out; }
video { height: 220px; border: 1px solid #ccc; background: #000; }
.videos { margin: 12px 0; }
.speed { display: inline-block; width: 12.5em; white-space: nowrap; text-align: center; font-weight: 600; color: #fff;
         background: #1a73e8; border-radius: 4px; padding: 1px 0; margin-bottom: 6px;
         font-variant-numeric: tabular-nums; }
.speeds button { margin: 0 2px 6px 0; padding: 2px 8px; border: 1px solid #1a73e8; background: #fff;
                 color: #1a73e8; border-radius: 4px; cursor: pointer; }
.speeds button.on { background: #1a73e8; color: #fff; }
.marks { margin-top: 6px; } .marks button, a.seek { font-size: 12px; margin: 2px 4px 2px 0; padding: 1px 6px;
  border: 1px solid #bbb; border-radius: 4px; background: #f8f8f8; cursor: pointer; color: #1a4fa0; }
.marks button.r-ok { border-color: #1e8e3e; background: #e6f4ea; color: #0d652d; }
.marks button.r-esc { border-color: #e37400; background: #fef3e0; color: #8a4b00; }
.marks button.r-bad { border-color: #d93025; background: #fce8e6; color: #a50e0e; }
.marks button.now { outline: 3px solid #1a73e8; outline-offset: 1px; font-weight: 700; }
.legend span { font-size: 11px; padding: 0 5px; border-radius: 3px; margin-left: 4px; border: 1px solid; }
.marks .turn { display: inline-flex; align-items: center; margin-right: 6px; }
.marks .turn button { margin-right: 1px; } .marks a.ev { font-size: 11px; color: #555; text-decoration: none; }
.marks a.ev:hover { text-decoration: underline; }
.event.flash { box-shadow: 0 0 0 3px #fbbc04; background: #fff8e1; transition: background 1.5s, box-shadow 1.5s; }
.event.fade { transition: background 1.5s, box-shadow 1.5s; }
.turnbadge { display: inline-block; font-size: 11px; font-weight: 700; color: #fff; background: #5f6368;
             border-radius: 4px; padding: 0 6px; margin-right: 6px; }
a.seek { text-decoration: none; margin-left: 6px; }
footer { margin-top: 28px; color: #555; }
.refresh { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 6px 12px; max-width: 980px; }
.guide { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px 14px; max-width: 980px; }
.guide dt { font-weight: 600; margin-top: 6px; } .guide dd { margin: 0 0 0 16px; color: #444; }
td.why { color: #555; font-size: 13px; }
details.fold { margin: 4px 0; } details.fold > summary { cursor: pointer; color: #444; }
details.fold > summary code { color: #222; }
table.sortable tr:first-child th { cursor: pointer; user-select: none; }
.filters { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 6px 12px; margin: 10px 0;
           max-width: 980px; }
.filters input[type=search] { margin: 0 10px 0 6px; padding: 2px 6px; width: 14em; }
details.pick { display: inline-block; position: relative; margin-right: 8px; }
details.pick > summary { cursor: pointer; padding: 2px 8px; border: 1px solid #bbb; border-radius: 4px;
                         background: #f8f8f8; list-style: none; white-space: nowrap; }
details.pick > div { position: absolute; z-index: 10; background: #fff; border: 1px solid #bbb; border-radius: 4px;
                     padding: 6px 10px; margin-top: 2px; max-height: 320px; overflow: auto; white-space: nowrap;
                     box-shadow: 0 2px 8px rgba(0,0,0,.15); }
details.pick > div label { display: block; padding: 1px 0; cursor: pointer; }
details.pick > div button { margin-top: 4px; }
.filters .shown { color: #555; margin-left: 6px; }
table.sortable th[data-dir=asc]::after { content: " \\25B2"; } table.sortable th[data-dir=desc]::after { content: " \\25BC"; }
"""

# Click a column header of a .sortable table to sort by it; click again to reverse. A cell's
# data-sort holds its raw value when the shown text is formatted ("3.5 min", "$0.0016").
# Numbers sort as numbers; empty cells and "–" stay last in both directions.
SORT_SCRIPT = """<script>
document.addEventListener("DOMContentLoaded", () => {
  const blank = key => key === "" || key === "\\u2013";
  const key = cell => (cell ? (cell.dataset.sort ?? cell.textContent) : "").trim();
  document.querySelectorAll("table.sortable").forEach(table => {
    const header = table.rows[0];
    [...header.cells].forEach((th, column) => th.addEventListener("click", () => {
      const dir = th.dataset.dir === "asc" ? "desc" : "asc";
      [...header.cells].forEach(other => delete other.dataset.dir);
      th.dataset.dir = dir;
      const sign = dir === "asc" ? 1 : -1;
      const rows = [...table.rows].slice(1);
      rows.sort((a, b) => {
        const x = key(a.cells[column]), y = key(b.cells[column]);
        if (blank(x) || blank(y)) return blank(x) - blank(y);
        const nx = Number(x), ny = Number(y);
        if (!isNaN(nx) && !isNaN(ny)) return sign * (nx - ny);
        return sign * x.localeCompare(y, undefined, {numeric: true});
      });
      rows.forEach(row => header.parentNode.appendChild(row));
    }));
  });
});
</script>"""


SCRIPT = "scripts/view_runs.py"
# Filters: a header cell with data-filter="Name" makes its column filterable; each cell
# lists its values in data-f, separated by "|". One bar (.filters) drives every
# table.filterable on the page. Each column gets a checkbox dropdown: checked values in a
# column combine with OR, columns with AND, nothing checked means all. A search box keeps
# rows whose text contains every typed word (case-insensitive).
FILTER_SCRIPT = """<script>
document.addEventListener("DOMContentLoaded", () => {
  const bar = document.querySelector(".filters");
  const tables = [...document.querySelectorAll("table.filterable")];
  if (!bar || !tables.length) return;
  const rows = () => tables.flatMap(t => [...t.rows].slice(1));
  const values = cell => (cell && cell.dataset.f ? cell.dataset.f.split("|") : []);
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search…";
  search.addEventListener("input", apply);
  bar.appendChild(search);
  const columns = [...tables[0].rows[0].cells]
    .map((th, index) => ({name: th.dataset.filter, index})).filter(c => c.name);
  const pickers = columns.map(column => {
    const seen = new Set();
    rows().forEach(r => values(r.cells[column.index]).forEach(v => seen.add(v)));
    const picker = document.createElement("details");
    picker.className = "pick";
    const summary = document.createElement("summary");
    const options = document.createElement("div");
    const boxes = [...seen].sort((a, b) => a.localeCompare(b, undefined, {numeric: true})).map(v => {
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = v;
      box.addEventListener("change", apply);
      label.append(box, " " + v);
      options.appendChild(label);
      return box;
    });
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "clear";
    clear.addEventListener("click", () => { boxes.forEach(b => { b.checked = false; }); apply(); });
    options.appendChild(clear);
    picker.append(summary, options);
    bar.appendChild(picker);
    return {column, summary, boxes};
  });
  const shown = document.createElement("span");
  shown.className = "shown";
  bar.appendChild(shown);
  // Close an open dropdown when clicking elsewhere.
  document.addEventListener("click", event => {
    document.querySelectorAll("details.pick[open]").forEach(d => { if (!d.contains(event.target)) d.open = false; });
  });
  function apply() {
    const words = search.value.toLowerCase().split(" ").filter(Boolean);
    let visible = 0, total = 0;
    pickers.forEach(({column, summary, boxes}) => {
      const chosen = boxes.filter(b => b.checked).map(b => b.value);
      summary.textContent = column.name + ": " + (chosen.length === 0 ? "all"
        : chosen.length === 1 ? chosen[0] : chosen.length + " selected");
    });
    rows().forEach(r => {
      const text = r.textContent.toLowerCase();
      const ok = words.every(w => text.includes(w)) && pickers.every(({column, boxes}) => {
        const chosen = boxes.filter(b => b.checked).map(b => b.value);
        return !chosen.length || values(r.cells[column.index]).some(v => chosen.includes(v));
      });
      r.hidden = !ok; total += 1; visible += ok;
    });
    shown.textContent = visible + " of " + total + " shown";
  }
  apply();
});
</script>"""

REPO = Path(__file__).resolve().parents[1]


def refresh_command(root):
    """The rebuild command, relative to the repository root (no machine-specific paths)."""
    try:
        target = root.relative_to(REPO).as_posix()
    except ValueError:
        target = root.name
    return f"python3 {SCRIPT} {target}"


def page(title, body, root):
    footer = f"<footer><p><small>Refresh: <code>{escape(refresh_command(root))}</code></small></p></footer>"
    return (f"<!doctype html><meta charset='utf-8'><title>{escape(title)}</title>"
            f"<style>{STYLE}</style>{SORT_SCRIPT}{FILTER_SCRIPT}<h1>{escape(title)}</h1>{body}{footer}")


def up(experiment, extra=0):
    """Relative link from an experiment directory (plus `extra` levels) back to the root index."""
    return "../" * (len(Path(experiment).parts) + extra) + "index.html"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_events(path):
    events = []
    try:
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return events


def fmt(value, digits=2):
    if value is None or value == "":
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def score_of(summary):
    """`harness report`'s (score, success) of one episode, None if it was dropped; a summary it cannot
    read (e.g. a reward that is not a number) counts as an unscored failure here instead of stopping
    every page."""
    try:
        return episode_score(summary)
    except (TypeError, ValueError, AttributeError, ArithmeticError):
        return 0.0, False


def scored(summaries):
    """`harness report`'s (score, success) of the episodes it scores; dropped ones are left out."""
    return [result for result in map(score_of, summaries) if result is not None]


def readable(summary):
    """A summary `harness report` itself can aggregate (load_episodes keeps those with a task and seed)."""
    try:
        episode_score(summary)
    except (TypeError, ValueError, AttributeError, ArithmeticError):
        return False
    return isinstance(summary.get("task"), str) and "seed" in summary


def layouts(events):
    """Layout source from the reset observation (the manifest does not record it)."""
    state = next((e.get("observation", {}).get("state", {}) for e in events if e.get("event") == "reset"), {})
    source = state.get("layout_source")
    if source == "eval":
        return f"EVAL pack {state.get('layout_pack_seed')}", "tag eval"
    if source:
        return "dev" if source == "procedural" else source, "tag"
    return "–", "tag"


def rung(runtime):
    """R0-R4 from the run's settings, with the driver's defaults (see configs/ablations/)."""
    if not runtime:
        return "–"
    if runtime.get("label"):  # a run that names itself instead of a rung
        return runtime["label"]
    layers = {"learned": False, "reactive": True, "handlers": True, **(runtime.get("layers") or {})}
    if runtime.get("mode") == "direct":
        return "R1" if layers["learned"] else "R0"
    key = (layers["learned"], layers["reactive"], layers["handlers"])
    name = {(False, False, True): "R2", (True, False, True): "R3", (True, True, True): "R4",
            (True, True, False): "R4 − handlers"}.get(key)
    if name is None:
        name = "program (" + ", ".join(k for k, on in layers.items() if on) + ")"
    if not runtime.get("verifier", True):
        name += " − verifier"
    # Skills a run left out because this machine lacked their services (configs/deploy).
    dropped = sorted(runtime.get("skills_dropped") or {})
    if dropped:
        name += " − " + ", ".join(dropped)
    return name


def setup(manifest):
    config = manifest.get("config", {})
    agent, runtime = config.get("main") or config.get("agent", {}), config.get("runtime", {})
    model = agent.get("model") or agent.get("backend") or "–"
    budget = config.get("max_model_calls") or config.get("rollout", {}).get("max_steps")
    return model, rung(runtime) if runtime else config.get("environment", {}).get("kind", "–"), budget


def collect(root):
    episodes = []
    # Experiments may sit in subfolders (e.g. runs/r0-models/<model>/<experiment>/seed-0).
    for summary_path in sorted(root.rglob("seed-*/summary.json")):
        directory = summary_path.parent
        # seed-<n>.error-<time> / .unstable-<time> are runs run_suite superseded; not episodes.
        if directory.parent == root or not directory.name.removeprefix("seed-").isdigit():
            continue
        summary = read_json(summary_path)
        manifest = read_json(directory / "manifest.json")
        events = read_events(directory / "events.jsonl")
        model, mode, budget = setup(manifest)
        source, source_class = layouts(events)
        experiment = directory.parent.relative_to(root).as_posix()
        episodes.append({"dir": directory, "experiment": experiment, "summary": summary,
                         "system_prompt": manifest.get("config", {}).get("system_prompt"),
                         "tools": manifest.get("tools"), "agent": manifest.get("config", {}).get("main") or manifest.get("config", {}).get("agent") or {},
                         "events": events, "model": model, "mode": mode, "budget": budget,
                         "layouts": source, "layouts_class": source_class})
    return episodes


def when(name):
    """20260930-004438-ca7092ce -> 2026-09-30 00:44."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.strptime(name[:15], "%Y%m%d-%H%M%S"))
    except ValueError:
        return name


def minutes(seconds):
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


ENDINGS = {"finished": "finished", "budget_exhausted": "out of budget", "error": "error",
           "episode_ended": "step limit", "interrupted": "interrupted", "success": "success"}


def ending(summary):
    """How one episode ended; an episode the simulator dropped as unstable reads "dropped"."""
    if score_of(summary) is None:
        return "dropped"
    return ENDINGS.get(summary.get("status"), str(summary.get("status")))


def ending_labels(summaries):
    return {ending(s) for s in summaries}


def data_f(values):
    """data-f attribute for the dropdown filters."""
    return f" data-f='{escape('|'.join(sorted(str(v) for v in values)))}'"


def endings(summaries):
    """How episodes ended, e.g. '2 finished, 1 out of calls'."""
    counts = {}
    for s in summaries:
        label = ending(s)
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{n} {label}" for label, n in sorted(counts.items(), key=lambda item: -item[1]))


def experiment_row(name, group):
    first, summaries = group[0], [e["summary"] for e in group]
    n = len(summaries)
    results = scored(summaries)
    counted, dropped = len(results), n - len(results)
    successes = sum(success for _, success in results)
    errors = sum(s.get("status") == "error" for s in summaries) - dropped
    provider, _, model = first["model"].rpartition("/")
    tasks = sorted({str(s.get("task")) for s in summaries})
    seeds = sorted({s.get("seed") for s in summaries}, key=str)
    cost = sum(s.get("cost_usd") or 0.0 for s in summaries)
    budget = f" / {first['budget']}" if first["budget"] else ""
    calls = sum(s.get("model_calls") or 0 for s in summaries) / n
    seconds = sum(s.get("wall_seconds") or 0 for s in summaries) / n
    folder = Path(name).parent.as_posix()
    return (
        f"<tr><td data-sort='{escape(Path(name).name)}'{data_f([Path(name).parts[0] if len(Path(name).parts) > 1 else '(top level)'])}>"
        f"<a href='{escape(name)}/index.html'>"
        f"{escape(when(Path(name).name))}</a>{f'<br><small>{escape(folder)}</small>' if folder != '.' else ''}</td>"
        f"<td{data_f([first['mode']])}><b>{escape(first['mode'])}</b></td>"
        f"<td data-sort='{escape(model)}'{data_f([first['model']])}>{escape(model)}<br><small>{escape(provider)}</small></td>"
        f"<td{data_f(tasks)}>{escape(', '.join(tasks) if len(tasks) <= 3 else f'{len(tasks)} tasks')}"
        f"<br><small>seeds {escape(', '.join(map(str, seeds)))}</small></td>"
        f"<td class=n data-sort='{successes / counted if counted else ''}'><span class={'ok' if successes else 'bad'}>"
        f"{successes} / {counted}</span>{f'<br><small>+{dropped} dropped</small>' if dropped else ''}</td>"
        f"<td class=n>{f'{100 * sum(score for score, _ in results) / counted:.1f}' if counted else '–'}</td>"
        f"<td{data_f(ending_labels(summaries))}>{escape(endings(summaries))}</td>"
        f"<td class=n data-sort='{calls}'>{calls:.1f}{budget}</td>"
        f"<td class=n data-sort='{cost if cost else ''}'>{'$' + format(cost, '.4f') if cost else '–'}</td>"
        f"<td class=n data-sort='{seconds}'>{minutes(seconds)}</td>"
        f"<td class=n>{errors or ''}</td></tr>")


RUNS_GUIDE = """
<details class=guide><summary><b>How to read this table</b></summary><dl>
<dt>Run</dt><dd>When the experiment started. One run is one <code>harness program</code> command: a set of
episodes (task × seed) with the same model and settings. Click it to see its episodes.</dd>
<dt>Rung</dt><dd>How much help the model gets. <b>R0</b>: it outputs one arm target per turn.
<b>R1</b>: R0 plus fixed skills and a learned policy. <b>R2</b>: it writes Python programs that call skills, with event
handlers and a verifier. <b>R3</b>: R2 plus the learned skill. <b>R4</b>: R3 plus reactive guards that run
every control tick. "− handlers" / "− verifier" are R4 with that part removed.</dd>
<dt>Model</dt><dd>The model that decides what to do, with its provider underneath.</dd>
<dt>Tasks · seeds</dt><dd>Which RoboDojo tasks were run. The seed picks the layout: which objects, and where.</dd>
<dt>Successes</dt><dd>Episodes the RoboDojo evaluator counted as complete, out of the scored episodes.
"+N dropped": episodes the simulator dropped as unstable; RoboDojo does not score them.</dd>
<dt>Score</dt><dd>RoboDojo's main metric, 0–100, as <code>harness report</code> computes it. A success counts
100; a failure counts its partial progress (the evaluator's process score). Episodes that crashed count 0.</dd>
<dt>Suites</dt><dd>A folder whose subfolders are runs of several tasks (e.g. a <code>harness run</code>
output) gets a row in the Suites table: Score / SR per capability dimension (the mean over its tasks) and overall
(the mean of the five dimensions), computed by <code>harness report</code>'s own code over the episodes of those
runs, as on RoboDojo's leaderboard. <b>Overall</b> is the benchmark number: it is shown only when every
task of every dimension has a scored episode, exactly as <code>harness report</code>. <b>Partial overall*</b>
is the same average (the mean of the five dimensions, each over its tasks) over only the tasks that have
scored episodes, with their count. It stands in when tasks are missing (e.g. a task that could not load, or
one whose episodes were all dropped) but is then not comparable to the leaderboard, which is why it is marked
with *. When every task is scored it reads "= Overall". A folder holding runs of different rungs or models pools them, as <code>harness report</code> does; the
row lists every rung and model it contains.</dd>
<dt>Ended</dt><dd><b>finished</b>: the model said it was done. <b>out of budget</b>: it used up its model
calls, time or cost limit. <b>step limit</b>: the task's simulation step limit ran out. <b>error</b>: the
episode crashed (see its page).</dd>
<dt>Calls / ep</dt><dd>Average model calls per episode, and the allowed maximum.</dd>
<dt>Cost</dt><dd>Total API cost of all episodes in the run.</dd>
<dt>Time / ep</dt><dd>Average wall-clock time per episode, including the model thinking and the robot moving.</dd>
<dt>Evaluation vs development</dt><dd><b>Evaluation</b> runs use RoboDojo's official layouts and are the
numbers to report. <b>Development</b> runs use layouts generated by this repo, never overlapping the official
ones; use them for trying things out. Their numbers are not comparable, so they are listed separately.</dd>
</dl></details>
"""

EPISODES_GUIDE = """
<details class=guide><summary><b>How to read this table</b></summary><dl>
<dt>Episode</dt><dd>One attempt at one task with one seed (layout). Click it for everything that happened.</dd>
<dt>Result</dt><dd>Whether the RoboDojo evaluator counted the task as complete.</dd>
<dt>Ended</dt><dd>Why the episode stopped: finished (the model said done), out of budget, step limit, or error.</dd>
<dt>Progress</dt><dd>The evaluator's partial credit, 0–100, when the task was not completed.</dd>
<dt>Calls</dt><dd>How many times the model was asked what to do next.</dd>
<dt>Sim steps</dt><dd>Simulation control steps used (25 per simulated second), against the task's step limit.</dd>
</dl></details>
"""

# summary.json fields in reading order: (key, label, explanation).
SUMMARY_FIELDS = [
    ("task", "Task", "The RoboDojo task."),
    ("seed", "Seed", "Picks the layout: which objects and where they are."),
    ("success", "Success", "Whether the RoboDojo evaluator counted the task as complete."),
    ("process_score", "Progress (0–100)", "The evaluator's partial credit when the task was not completed."),
    ("reward", "Reward", "1 on success, otherwise progress / 100. This is what the score averages."),
    ("status", "Ended", "Why the episode stopped."),
    ("error", "Error", "The error message, if the episode crashed."),
    ("model_calls", "Model calls", "Times the model was asked what to do next."),
    ("verifier_calls", "Verifier calls", "Times a separate model call checked a claim about the scene (R2 and up)."),
    ("cells", "Actions / programs run", "How many of the model's replies were executed on the robot."),
    ("steps", "Simulation steps", "Control steps used, 25 per simulated second."),
    ("input_tokens", "Input tokens", "Text and images sent to the model, over all calls."),
    ("output_tokens", "Output tokens", "Text the model wrote, over all calls."),
    ("cost_usd", "Cost (USD)", "API cost of all model and verifier calls."),
    ("agent_seconds", "Model time", "Time spent waiting for the model to reply."),
    ("kernel_seconds", "Robot time", "Time spent executing the model's actions in the simulator."),
    ("wall_seconds", "Total time", "Wall-clock time for the whole episode."),
]


def summary_value(key, value):
    if key.endswith("_seconds") and isinstance(value, (int, float)):
        return minutes(value)
    if key == "status":
        return endings([{"status": value}]).split(" ", 1)[1] if value else "–"
    if key == "cost_usd" and isinstance(value, (int, float)):
        return f"${value:.4f}"
    if key == "success":
        return "✅ yes" if value else "❌ no"
    if key in ("input_tokens", "output_tokens") and isinstance(value, int):
        return f"{value:,}"
    return fmt(value, 2)


def section_of(group):
    source = group[0]["layouts"]
    return ("Evaluation layouts (official packs)" if source.startswith("EVAL") else
            "Development layouts" if source == "dev" else "Other (mock runs, or no layout recorded)")


def suites_html(experiments):
    """`harness report`'s Score / SR by dimension for each folder whose subfolders are runs of
    several tasks (e.g. a run_suite output): aggregate() over the episodes of those runs."""
    suites = {}
    for name, group in experiments.items():
        if len(Path(name).parts) > 1:
            suites.setdefault(Path(name).parent.as_posix(), []).extend(group)
    rows = []
    for folder, group in sorted(suites.items()):
        # As harness report's load_episodes, skip summaries without a task or seed (and any it could not read).
        total = len(group)
        group = [e for e in group if readable(e["summary"])]
        summaries = [e["summary"] for e in group]
        skipped = total - len(group)
        if len({s.get("task") for s in summaries}) < 2:
            continue
        try:
            result = aggregate(summaries)
        except (TypeError, ValueError, AttributeError, ArithmeticError, KeyError):
            # A summary harness report itself cannot aggregate: no numbers rather than no pages.
            result = {"dimensions": {d: {"score": None, "sr": None, "tasks": 0, "of": len(DIMENSIONS[d])}
                                     for d in DIMENSIONS}, "overall": None}
            skipped = total
        cell = lambda d: ("–" if d.get("score") is None else f"{d['score']:.1f} / {d['sr']:.1f}")
        dims = "".join(f"<td class=n>{cell(result['dimensions'][d])}<br><small>{result['dimensions'][d]['tasks']}"
                       f"/{result['dimensions'][d]['of']} tasks</small></td>" for d in DIMENSIONS)
        overall = result["overall"] or {}
        partial = partial_overall(result["dimensions"])
        rung, model = {e["mode"] for e in group}, {e["model"] for e in group}
        cost = sum(s.get("cost_usd") or 0.0 for s in summaries)
        note = f"<br><small>{skipped} unreadable skipped</small>" if skipped else ""
        rows.append(f"<tr><td><b>{escape(folder)}</b>{note}</td><td>{escape(', '.join(sorted(rung)))}</td>"
                    f"<td>{escape(', '.join(sorted(model)))}</td>{dims}<td class=n><b>{cell(overall)}</b></td>"
                    f"<td class=n data-sort='{'' if partial is None else partial['score']}'>{partial_cell(partial)}</td>"
                    f"<td class=n>{'$' + format(cost, '.2f') if cost else '–'}</td></tr>")
    if not rows:
        return ""
    head = ("<table class=sortable><tr><th>Suite</th><th>Rung</th><th>Model</th>"
            + "".join(f"<th>{escape(d)}<br><small>Score / SR</small></th>" for d in DIMENSIONS)
            + "<th>Overall<br><small>Score / SR<br>all 42 tasks</small></th>"
            + "<th>Partial overall*<br><small>Score / SR<br>tasks scored only</small></th><th>Cost</th></tr>")
    return (f"<h2>Suites · {len(rows)}</h2><p>Score / SR as <code>harness report</code> computes them, over the "
            f"episodes of the folder's task runs. <b>Overall</b> is the benchmark number and needs every task; "
            f"<b>Partial overall*</b> is the same average over only the tasks that have scored episodes, so it is "
            f"not comparable to the leaderboard when tasks are missing.</p>{head}{''.join(rows)}</table>")


def partial_overall(dimensions):
    """The overall average over the tasks with scored episodes: the mean of the five dimensions, each over its
    scored tasks. None unless every dimension has at least one scored task. Unlike `harness report`'s
    overall it does not need every task, so it is marked and reported with its task count."""
    present = [dimensions[d] for d in DIMENSIONS]
    if any(d["score"] is None for d in present):
        return None
    mean = lambda key: sum(d[key] for d in present) / len(present)
    return {"score": mean("score"), "sr": mean("sr"), "tasks": sum(d["tasks"] for d in present),
            "of": sum(d["of"] for d in present)}


def partial_cell(partial):
    if partial is None:
        return "–<br><small>a dimension has no task scored</small>"
    if partial["tasks"] == partial["of"]:  # every task scored: this is the Overall itself
        return f"{partial['score']:.1f} / {partial['sr']:.1f}<br><small>= Overall</small>"
    return f"{partial['score']:.1f}* / {partial['sr']:.1f}*<br><small>{partial['tasks']}/{partial['of']} tasks</small>"


def index_html(root, episodes):
    experiments = {}
    for episode in episodes:
        experiments.setdefault(episode["experiment"], []).append(episode)
    sections = {"Evaluation layouts (official packs)": [], "Development layouts": [], "Other (mock runs, or no layout recorded)": []}
    for name in sorted(experiments, key=lambda name: Path(name).name, reverse=True):
        sections[section_of(experiments[name])].append(experiment_row(name, experiments[name]))
    head = ("<table class='sortable filterable'><tr><th data-filter=Sweep>Run</th><th data-filter=Rung>Rung</th>"
            "<th data-filter=Model>Model</th><th data-filter=Task>Tasks</th><th>Successes</th>"
            "<th>Score</th><th data-filter=Ended>Ended</th><th>Calls / ep<br><small>of budget</small></th><th>Cost</th>"
            "<th>Time / ep</th><th>Errors</th></tr>")
    body = suites_html(experiments) + "".join(f"<h2>{escape(title)} · {len(rows)}</h2>" + head + "".join(rows)
                                              + "</table>" for title, rows in sections.items() if rows)
    note = (f"<p>{len(experiments)} runs · {len(episodes)} episodes · "
            f"generated {time.strftime('%Y-%m-%d %H:%M')}</p>")
    refresh = (f"<p class=refresh><b>Refresh</b> (from the repo root): "
               f"<code>{escape(refresh_command(root))}</code></p>")
    return page("Runs", note + refresh + RUNS_GUIDE + "<div class=filters><b>Filter:</b> </div>" + body, root)


def experiment_html(root, name, group):
    first = group[0]
    rows = []
    for episode in group:
        s = episode["summary"]
        counted = score_of(s)
        label = "dropped" if counted is None else "success" if counted[1] else "fail"
        result = f"<span class={'ok' if label == 'success' else 'bad'}>{label}</span>"
        rows.append(
            f"<tr><td><a href='{escape(episode['dir'].name)}/view.html'>{escape(episode['dir'].name)}</a></td>"
            f"<td{data_f([s.get('task')])}>{escape(fmt(s.get('task')))}</td>"
            f"<td{data_f([label])}>{result}</td>"
            f"<td{data_f(ending_labels([s]))}>{escape(endings([s]))}</td>"
            f"<td class=n>{fmt(s.get('process_score'), 1)}</td><td class=n>{fmt(s.get('model_calls'))}</td>"
            f"<td class=n>{fmt(s.get('steps'))}</td><td class=n>{fmt(s.get('cost_usd'), 4)}</td>"
            f"<td class=n data-sort='{s.get('wall_seconds') or 0}'>{minutes(s.get('wall_seconds') or 0)}</td>"
            f"<td>{escape(fmt(s.get('error')))}</td></tr>")
    header = (f"<p><a href='{up(name)}'>← all runs</a> · <span class='{first['layouts_class']}'>"
              f"{escape(first['layouts'])}</span> · {escape(first['mode'])} · {escape(first['model'])} · "
              f"{escape(name)}</p>")
    table = ("<div class=filters><b>Filter:</b> </div><table class='sortable filterable'><tr><th>Episode</th>"
             "<th data-filter=Task>Task</th><th data-filter=Result>Result</th><th data-filter=Ended>Ended</th>"
             "<th>Progress</th>"
             "<th>Calls</th><th>Sim steps</th><th>Cost USD</th><th>Time</th><th>Error</th></tr>"
             + "".join(rows) + "</table>")
    return page(f"{when(Path(name).name)} · {first['mode']} · {first['model']}",
                header + EPISODES_GUIDE + table, root)


CONTROL_RATE = 25  # RoboDojo control steps per second (arx_x5 collect_freq): 1 video second


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def turn_spans(events):
    """(start, end, thought) per executed action: video seconds from the control step it ended
    on, and the model's reply time before it (the video does not show it: the simulator waits).

    The motion clip starts at reset and plays at CONTROL_RATE frames per second (new clips
    have one frame per control step; older 250 fps clips play in the same real time).
    """
    spans, previous, thought = {}, 0, None
    for event in events:
        if event.get("event") == "model_reply":
            thought = event.get("seconds")
        if event.get("event") != "cell":
            continue
        step = ((event.get("result") or {}).get("state") or {}).get("step")
        if not isinstance(step, (int, float)):
            continue
        spans[id(event)] = (previous / CONTROL_RATE, step / CONTROL_RATE, thought)
        previous, thought = step, None
    return spans


def turn_numbers(events):
    """Turn number of each event in a turn: the request, reply and execution of the n-th
    executed action. A call whose reply executed nothing gets no number."""
    numbers, pending, turn = {}, [], 0
    for event in events:
        kind = event.get("event")
        if kind == "model_request":
            pending = [event]
        elif kind in ("model_reply", "protocol_error") and pending:
            pending.append(event)
        elif kind == "cell":
            turn += 1
            for item in pending + [event]:
                numbers[id(item)] = turn
            pending = []
    return numbers


# How a program cell can end (runtime/kernel.py). It ends normally when it returns (ok), when
# the program calls finish() (finished) or when the environment ends the episode
# (episode_ended); an escalation hands control back to the model; anything else failed.
NORMAL_ENDINGS = {"ok", "finished", "episode_ended"}


def result_class(event):
    """Marker color by how the action ended: normal endings green, escalated amber, anything else red."""
    status = (event.get("result") or {}).get("status")
    return ("r-ok" if status in NORMAL_ENDINGS else "r-esc" if status == "escalated" else "r-bad"), str(status)


def action_label(event):
    try:
        action = json.loads(event.get("action") or "")
    except ValueError:
        return "program"
    return " ".join(str(v) for v in (action.get("type"), action.get("arm", "").replace("_arm", "")) if v)


VIDEO_SPEED = 1  # videos play in real time (one frame per control step, or labeled 250 fps)
SPEEDS = (0.1, 0.5, 1, 3, 5, 10)  # Chrome supports 0.0625x to 16x

# Speed buttons for a video block: set every clip in the block, and keep the label in step
# with the rate, including changes made from a player's own menu.
SPEED_SCRIPT = """<script>
function setSpeed(block, rate) {
  block.dataset.rate = rate;
  block.querySelectorAll("video").forEach(v => { v.defaultPlaybackRate = v.playbackRate = rate; });
  showSpeed(block, rate);
}
// Highlight every card of turn n for a few seconds (after the ↓ link scrolls to it).
function flashTurn(n) {
  const cards = document.querySelectorAll(".event[data-turn='" + n + "']");
  cards.forEach(c => { c.classList.remove("fade"); c.classList.add("flash"); });
  setTimeout(() => cards.forEach(c => { c.classList.add("fade"); c.classList.remove("flash"); }), 2500);
}
// Move every camera to time t (seconds of video). The page does not scroll.
function seekVideos(t) {
  document.querySelectorAll(".videos video").forEach(v => { v.currentTime = t; });
  highlightTurn();
}
// Highlight the turn being shown. Runs every frame while playing, so it stays right at
// any speed (timeupdate fires only ~4 times a second, too coarse at 3x-10x).
function highlightTurn() {
  const video = document.querySelector(".videos video");
  if (!video) return;
  const t = video.currentTime;
  document.querySelectorAll(".marks button").forEach(b =>
    b.classList.toggle("now", t >= +b.dataset.start && t < +b.dataset.end));
}
// The cameras are one clip: playing, pausing or dragging any of them does the same to
// all. The camera the user last touched is the clock the others follow.
document.addEventListener("DOMContentLoaded", () => {
  const videos = [...document.querySelectorAll(".videos video")];
  if (!videos.length) return;
  let leader = videos[0];
  const follow = () => videos.forEach(v => {
    if (v !== leader && Math.abs(v.currentTime - leader.currentTime) > 0.05) v.currentTime = leader.currentTime;
  });
  const tick = () => { highlightTurn(); follow(); if (!leader.paused && !leader.ended) requestAnimationFrame(tick); };
  videos.forEach(video => {
    video.addEventListener("play", () => {
      leader = video;
      videos.forEach(v => { if (v.paused) v.play(); });
      requestAnimationFrame(tick);
    });
    video.addEventListener("pause", () => {
      if (video.ended) return;
      videos.forEach(v => { if (!v.paused) v.pause(); });
    });
    // Dragging the timeline: the dragged camera leads, the others jump to it. Programmatic
    // seeks land within 0.05 s of the leader, so they do not bounce back.
    video.addEventListener("seeking", () => {
      if (Math.abs(video.currentTime - leader.currentTime) > 0.05) leader = video;
      follow();
      highlightTurn();
    });
  });
  highlightTurn();
});
function applySpeed(video) {
  const block = video.closest(".videos");
  video.defaultPlaybackRate = video.playbackRate = +block.dataset.rate;
}
function showSpeed(block, rate) {
  block.querySelector(".speed").textContent = "\u25B6 plays at " + rate + "\u00D7 speed";
  block.querySelectorAll(".speeds button").forEach(b => b.classList.toggle("on", +b.dataset.rate === rate));
}
</script>"""


def video_html(src, camera, directory=None):
    """One camera clip, played at VIDEO_SPEED times real time (the player's menu can change it).

    With the episode directory, the file's modification time is added to the URL, so a
    browser never plays a stale cached copy after the file is rewritten.
    """
    path = Path(directory) / src if directory is not None else None
    if path is not None and path.is_file():
        src = f"{src}?v={int(path.stat().st_mtime)}"
    return (f"<figure><video src='{escape(src)}' controls muted preload=metadata "
            f"onloadedmetadata='applySpeed(this)' "
            f"onratechange='showSpeed(this.closest(\".videos\"), this.playbackRate)'></video>"
            f"<figcaption>{escape(camera)}</figcaption></figure>")


def videos_block(title, clips, marks=""):
    note = (f"<div class=speed>▶ plays at {VIDEO_SPEED}× speed</div> "
            "<small>Recorded in real time. Speed applies to all cameras:</small> <span class=speeds>"
            + "".join(f"<button data-rate={r:g} class='{'on' if r == VIDEO_SPEED else ''}' "
                      f"onclick='setSpeed(this.closest(\".videos\"), {r:g})'>{r:g}×</button>" for r in SPEEDS)
            + "</span>")
    return (f"<div class=videos data-rate={VIDEO_SPEED:g}><h2>{escape(title)}</h2>{note}<br>"
            f"{''.join(clips)}{marks}</div>")


def _harness_api():
    """Names a program can call: the lowercase names the skills export (skills/*/skill.toml),
    so the list follows the code."""
    import tomllib

    names = set()
    for path in sorted((Path(__file__).resolve().parents[1] / "skills").glob("*/skill.toml")):
        try:
            exports = tomllib.loads(path.read_text()).get("exports", [])
        except (OSError, tomllib.TOMLDecodeError):
            continue
        names |= {name for name in exports if name.isidentifier() and name.islower() and not name.startswith("_")}
    return frozenset(names)


HARNESS_API = _harness_api()
STRING_TOKENS = {tokenize.STRING} | {getattr(tokenize, name) for name in ("FSTRING_START", "FSTRING_MIDDLE",
                                                                          "FSTRING_END") if hasattr(tokenize, name)}
LONG_STRING = 100  # characters; longer one-line strings show their start and expand on click


def code_segments(code):
    """(text, css class or None) pieces whose texts join back to `code` exactly: keywords, strings,
    numbers, comments and calls of harness skills are marked. Code that does not tokenize is shown
    unmarked."""
    # tokenize counts lines at "\n" only (unlike str.splitlines, which also splits at \r, \x0c, ...).
    starts, offset = [], 0
    for line in code.split("\n"):
        starts.append(offset)
        offset += len(line) + 1
    segments, position = [], 0
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                  if t.type not in (tokenize.ENDMARKER, tokenize.DEDENT)]
        for index, token in enumerate(tokens):
            start = starts[token.start[0] - 1] + token.start[1]
            end = starts[token.end[0] - 1] + token.end[1]
            if start < position or end < start or end > len(code):
                continue
            kind = None
            if token.type == tokenize.NAME and keyword.iskeyword(token.string):
                kind = "k"
            elif token.type == tokenize.NAME and token.string in HARNESS_API:
                # A call of the skill itself: `pick(`, not `obs.pick(` or a variable named after one.
                following = next((t for t in tokens[index + 1:] if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
                before = tokens[index - 1].string if index else ""
                kind = "a" if following is not None and following.string == "(" and before != "." else None
            elif token.type in STRING_TOKENS:
                kind = "s"
            elif token.type == tokenize.NUMBER:
                kind = "n"
            elif token.type == tokenize.COMMENT:
                kind = "c"
            if start > position:
                segments.append((code[position:start], None))
            segments.append((code[start:end], kind))
            position = end
    except Exception:  # anything tokenize cannot read is shown unmarked, never breaks the page
        segments, position = [], 0
    if position < len(code):
        segments.append((code[position:], None))
    return segments


def code_html(code):
    """A program as a numbered, highlighted code block; long one-line strings show their start
    and the full text on click."""
    lines, current = [], []
    for text, kind in code_segments(code):
        for i, part in enumerate(text.split("\n")):
            if i:
                lines.append("".join(current))
                current = []
            if not part:
                continue
            if kind == "s" and len(part) > LONG_STRING and len(text.split("\n")) == 1:
                current.append(f"<span class='s x' title='click to show the whole string' "
                               f"data-full='{escape(part, quote=True)}' onclick='this.textContent=this.dataset.full;"
                               f"this.classList.remove(\"x\")'>{escape(part[:LONG_STRING - 20])}"
                               f"…{escape(part[-1])} <small>({len(part)} chars)</small></span>")
            elif kind:
                current.append(f"<span class={kind}>{escape(part)}</span>")
            else:
                current.append(escape(part))
    if current or not lines:
        lines.append("".join(current))
    return "<pre class=code>" + "".join(f"<span class=l>{line or ' '}</span>" for line in lines) + "</pre>"


def code_outline(code):
    """The program at a glance: the harness skills it calls (with line numbers) and its top-level
    comments as steps. Empty when the code does not parse."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. invalid escape sequences in the model's strings
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in HARNESS_API:
            calls.setdefault(node.func.id, []).append(node.lineno)
    order = sorted(calls, key=lambda name: min(calls[name]))
    skills = " · ".join(f"<b>{escape(name)}</b> ×{len(calls[name])} "
                        f"<small>(L{', L'.join(str(n) for n in sorted(calls[name]))})</small>" for name in order)
    steps = [(number, line.strip()[1:].strip()) for number, line in enumerate(code.split("\n"), 1)
             if line.startswith("#") and line.strip()[1:].strip()]
    out = f"<div>Skill calls: {skills}</div>" if skills else ""
    if len(steps) > 1:
        out += "<div>Steps (its comments):<ol>" + "".join(
            f"<li>{escape(text)} <small>(L{number})</small></li>" for number, text in steps) + "</ol></div>"
    return f"<div class=outline>{out}</div>" if out else ""


def reply_html(text):
    """A model reply. A program reply is read as the harness reads it (runtime/protocol.py
    parse_reply): its thought as text and its program, as the kernel ran it, outlined and
    highlighted, with the raw reply one click away. Other replies are shown as they are."""
    if "<python>" not in text or "<action>" in text:  # direct-control replies follow their own format
        return f"<pre>{escape(text)}</pre>"
    raw = fold(f"Raw reply, exactly as the model returned it ({len(text):,} characters)", f"<pre>{escape(text)}</pre>")
    try:
        reply = parse_reply(text)
    except ProtocolError as error:
        return f"<p class=bad>The harness rejected this reply: {escape(str(error))}</p><pre>{escape(text)}</pre>"
    if reply.kind != "python":
        return f"<pre>{escape(text)}</pre>"
    thought = f"<pre class=prose>{escape(reply.thought)}</pre>" if reply.thought else ""
    rest = text[text.index("</python>", text.index("<python>")) + len("</python>"):].strip()
    after = f"<pre class=prose>{escape(rest)}</pre>" if rest else ""
    return thought + code_outline(reply.body) + code_html(reply.body) + after + raw


def fold(label, inner):
    """Hidden by default, one click to open."""
    return f"<details class=fold><summary>{label}</summary>{inner}</details>"


def first_line(text, limit=110):
    line = text.strip().split("\n", 1)[0]
    return line if len(line) <= limit else line[:limit] + "…"


def parts_html(content):
    if isinstance(content, str):
        return f"<pre>{escape(content)}</pre>"
    out, figures, camera = [], [], ""
    for part in content or []:
        if part.get("type") == "input_image":
            figures.append(f"<figure><img tabindex=0 src='{escape(part.get('image_url', ''))}'>"
                           f"<figcaption>{escape(camera)}</figcaption></figure>")
        elif part.get("type") == "input_text":
            text = part.get("text", "")
            if text.startswith("Camera: "):
                camera = text.removeprefix("Camera: ")
            else:
                out.append(fold(f"<code>{escape(first_line(text))}</code>", f"<pre>{escape(text)}</pre>"))
    return "".join(out) + "".join(figures)


EVENT_NAMES = {
    "reset": "Episode start: instruction and first camera images",
    "model_request": "Sent to the model",
    "model_reply": "Model reply",
    "cell": "Executed on the robot",
    "verify": "Verifier check",
    "finish": "Model declared the task done",
    "outcome": "Evaluator result",
    "episode_video": "Episode video",
    "protocol_error": "Reply could not be used (wrong format)",
    "malformed_dropped": "Unusable replies removed from the model's history",
    "context_shortened": "History shortened to fit the model",
    "kernel_died": "Program process crashed",
    "error": "Episode error",
    "cleanup_error": "Problem while closing the episode",
    "interrupted": "Episode interrupted",
    "library_loaded": "Learned library loaded",
}


BAD_EVENTS = {"protocol_error", "kernel_died", "error", "cleanup_error", "interrupted", "malformed_dropped"}


def event_class(event):
    """Left-edge color: gray input, blue reply, green/red result, red problems."""
    kind = event.get("event")
    if kind == "model_request":
        return "e-input"
    if kind == "model_reply":
        return "e-reply"
    if kind == "cell":
        status = (event.get("result") or {}).get("status")
        return "e-ok" if status in NORMAL_ENDINGS else "e-esc" if status == "escalated" else "e-bad"
    if kind == "outcome":
        return "e-verdict " + ("e-ok" if (event.get("outcome") or {}).get("success") else "e-bad")
    if kind == "finish":
        return "e-reply"
    if kind in BAD_EVENTS:
        return "e-bad"
    return "e-frame"


def event_html(event, span=None, turn=None):
    html = block_html(event, span)
    turn_attribute = ""
    if turn is not None:  # same number as the turn's marker under the video
        html = html.replace("<h3>", f"<h3><span class=turnbadge>Turn {turn}</span>", 1)
        turn_attribute = f" data-turn={turn}"
    return html.replace("<div class=event>",
                        f"<div class='event {event_class(event)}' id='ev-{event.get('seq')}'{turn_attribute}>", 1)


def context_row(index, row):
    """One message of a call's full context: role, size, images kept (and omitted)."""
    images = str(row.get("images", 0)) + (f" (+{row['omitted']} omitted)" if row.get("omitted") else "")
    return (f"<tr><td>{index}</td><td>{escape(str(row.get('role')))}</td><td>{row.get('chars', 0):,}</td>"
            f"<td>{images}</td><td><code>{escape(row.get('text', ''))}</code></td></tr>")


def block_html(event, span=None):
    kind = event.get("event")
    head = f"<h3>{escape(EVENT_NAMES.get(kind, str(kind)))}"
    if kind == "reset":
        observation = event.get("observation", {})
        images = "".join(f"<figure><img tabindex=0 src='{escape(url)}'><figcaption>{escape(name)}</figcaption></figure>"
                         for name, url in (observation.get("images") or {}).items())
        return (f"<div class=event>{head}</h3><pre>{escape(str(observation.get('instruction')))}</pre>"
                f"{images}</div>")
    if kind == "model_request":
        messages = [m for m in event.get("new_messages", []) if m.get("role") != "assistant"]
        body = "".join(parts_html(m.get("content")) for m in messages)
        context = event.get("context") or []
        if context:
            images = sum(row.get("images", 0) for row in context)
            omitted = sum(row.get("omitted", 0) for row in context)
            rows = "".join(context_row(i, row) for i, row in enumerate(context))
            body += fold(f"Full context of this call: {len(context)} messages · {images} images"
                         f"{f' · {omitted} older images omitted' if omitted else ''}",
                         "<table class=context><tr><th>#</th><th>role</th><th>characters</th><th>images</th>"
                         f"<th>opening text</th></tr>{rows}</table>")
        return f"<div class=event>{head} · call {event.get('call', 0) + 1}</h3>{body}</div>"
    if kind == "model_reply":
        usage = event.get("usage") or {}
        thinking = f" ({usage['reasoning_tokens']} reasoning)" if usage.get("reasoning_tokens") else ""
        meta = (f"{fmt(event.get('seconds'), 1)} s · {usage.get('input_tokens')} in / "
                f"{usage.get('output_tokens')}{thinking} out tokens · ${fmt(usage.get('cost'), 5)}")
        reasoning = event.get("reasoning")
        extra = fold(f"Reasoning, as returned by the provider ({len(reasoning):,} characters)",
                     f"<pre>{escape(reasoning)}</pre>") if reasoning else ""
        return f"<div class=event>{head} · {meta}</h3>{reply_html(str(event.get('text')))}{extra}</div>"
    if kind == "cell":
        result = event.get("result") or {}
        status = result.get("status")
        tone = "ok" if status in NORMAL_ENDINGS else "esc" if status == "escalated" else "bad"
        status_html = f"<span class={tone}>{escape(str(status))}</span>"
        executions = "".join(
            f"<li>{escape(str(item.get('skill')))}: {escape(str((item.get('event') or {}).get('kind')))} "
            f"({item.get('ticks')} ticks) {escape(json.dumps((item.get('event') or {}).get('detail') or {}))}</li>"
            for item in result.get("events") or [])
        if event.get("code") and not event.get("action"):
            # The model's reply above shows this program in full; here it is one click away.
            # (Direct-control cells carry the model's action, shown as before.)
            code = str(event["code"])
            body = fold(f"Program: {len(code.splitlines())} lines (the code in the reply above)", code_html(code))
        else:
            body = f"<pre>{escape(str(event.get('action')))}</pre>"
        if result.get("error"):
            body += f"<pre class=bad>{escape(str(result['error']))}</pre>"
        details = (f"<pre>{escape(result['stdout'])}</pre>" if result.get("stdout") else "") + (
            f"<ul>{executions}</ul>" if executions else "")
        if details:
            body += fold("Output and skill results", details)
        seek = (f"<a class=seek href='javascript:void(0)' onclick='seekVideos({span[0]:.2f})'>"
                f"▶ {clock(span[0])}–{clock(span[1])}</a>") if span else ""
        return (f"<div class=event>{head} · {status_html} · {fmt(result.get('seconds'), 1)} s{seek}</h3>"
                f"{body}</div>")
    if kind == "episode_video":
        return videos_block(EVENT_NAMES["episode_video"],
                            [video_html(path, camera) for camera, path in (event.get("files") or {}).items()])
    rest = {k: v for k, v in event.items() if k not in ("seq", "time", "event")}
    if kind == "outcome":
        o = rest.get("outcome") or {}
        line = (f"{'✅ success' if o.get('success') else '❌ not completed'} · progress {fmt(o.get('process_score'), 1)}"
                f" · {fmt(o.get('steps'))} simulation steps")
    else:
        line = escape(first_line(str(rest.get("message") or rest.get("text") or json.dumps(rest))))
    return (f"<div class=event>{head}</h3><p>{line}</p>"
            f"{fold('Raw event', f'<pre>{escape(json.dumps(rest, indent=1))}</pre>')}</div>")


def episode_html(root, episode):
    s = episode["summary"]
    known = {key for key, _label, _why in SUMMARY_FIELDS}
    main = {"task", "seed", "success", "process_score", "status", "error", "model_calls", "cost_usd", "wall_seconds"}

    def row(key, label, why):
        return (f"<tr><th>{escape(label)}</th><td>{escape(summary_value(key, s[key]))}</td>"
                f"<td class=why>{escape(why)}</td></tr>")

    fields = [(k, label, why) for k, label, why in SUMMARY_FIELDS if k in s and not (k == "error" and not s[k])]
    rows = "".join(row(*f) for f in fields if f[0] in main)
    more = "".join(row(*f) for f in fields if f[0] not in main) + "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(fmt(value, 4))}</td><td></td></tr>"
        for key, value in s.items() if key not in known and key != "run")
    header = (f"<p><a href='{up(episode['experiment'], 1)}'>← all runs</a> · <a href='../index.html'>← this run "
              f"({escape(when(Path(episode['experiment']).name))})</a> · "
              f"<span class='{episode['layouts_class']}'>{escape(episode['layouts'])}</span> · "
              f"{escape(episode['model'])} · {escape(episode['mode'])}</p><table>{rows}</table>"
              f"{fold('More details: tokens, steps, time split', f'<table>{more}</table>') if more else ''}")
    video_events = [e for e in episode["events"] if e.get("event") == "episode_video"]
    files = (video_events[-1].get("files") or {}) if video_events else {}
    clips = [video_html(path, camera, episode["dir"]) for camera, path in files.items()]
    if not clips:
        # Videos placed in the folder by hand (no episode_video event).
        clips = [video_html(f"videos/{v.name}", v.stem.removeprefix("episode-"), episode["dir"])
                 for v in sorted((episode["dir"] / "videos").glob("*.mp4"))]
    spans = turn_spans(episode["events"]) if clips else {}
    turns = turn_numbers(episode["events"])
    body = "".join(event_html(e, spans.get(id(e)), turns.get(id(e)))
                   for e in episode["events"] if e.get("event") != "episode_video")
    if episode["system_prompt"]:
        # Recorded once in manifest.json; sent as the system message with every model call.
        prompt = episode["system_prompt"]
        body = (f"<div class='event e-frame'><h3>System prompt · sent with every model call · "
                f"{len(prompt):,} characters</h3>"
                f"{fold(f'<code>{escape(first_line(prompt))}</code>', f'<pre>{escape(prompt)}</pre>')}</div>" + body)
    if episode["tools"]:
        # Function definitions sent with every call, for harnesses that use tool calling;
        # R0-R4 describe their actions in the prompt instead.
        tools = episode["tools"]
        names = ", ".join(str((tool.get("function") or tool).get("name")) for tool in tools)
        body = (f"<div class='event e-frame'><h3>Tools · sent with every model call · {escape(names)}</h3>"
                f"{fold('<code>function definitions</code>', f'<pre>{escape(json.dumps(tools, indent=1))}</pre>')}"
                "</div>" + body)
    if episode["agent"]:
        settings = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(json.dumps(v))}</td></tr>"
                           for k, v in episode["agent"].items())
        body = (f"<div class='event e-frame'><h3>Model settings · {escape(str(episode['agent'].get('model', '')))}</h3>"
                f"{fold('<code>request settings</code>', f'<table>{settings}</table>')}</div>" + body)
    s = episode["summary"]
    title = f"{fmt(s.get('task'))} · seed {fmt(s.get('seed'))} · {episode['mode']} · {episode['model']}"
    cells = [e for e in episode["events"] if id(e) in spans]
    # Each turn's first message (what was sent to the model) is where "events" jumps to.
    first, request = {}, None
    for e in episode["events"]:
        if e.get("event") == "model_request":
            request = e.get("seq")
        elif id(e) in spans:
            first[id(e)] = request if request is not None else e.get("seq")
            request = None
    marks = ("<div class=marks><small>Jump to turn (the video shows robot motion only; the simulator "
             "waits while the model thinks). ↓ opens that turn's messages below.</small> <small class=legend>"
             "<span style='border-color:#1e8e3e;background:#e6f4ea'>ok / finished / ended</span>"
             "<span style='border-color:#e37400;background:#fef3e0'>escalated</span>"
             "<span style='border-color:#d93025;background:#fce8e6'>error</span>"
             "<span style='border-color:#bbb;outline:2px solid #1a73e8'>playing</span></small><br>" + "".join(
        f"<span class=turn><button class={result_class(e)[0]} title='{escape(result_class(e)[1])}' "
        f"data-start={spans[id(e)][0]:.2f} data-end={spans[id(e)][1]:.2f} "
        f"onclick='seekVideos({spans[id(e)][0]:.2f})'>{n} · {escape(action_label(e))}"
        f"{f' · thought {spans[id(e)][2]:.1f} s' if isinstance(spans[id(e)][2], (int, float)) else ''}"
        f" · ▶ {clock(spans[id(e)][0])}</button><a class=ev href='#ev-{first[id(e)]}' onclick='flashTurn({n})' "
        f"title='Jump to this turn in the events'>↓</a></span>"
        for n, e in ((turns[id(e)], e) for e in cells)) + "</div>") if cells else ""
    video = videos_block("Video", clips, marks) if clips else ""
    return page(title, (SPEED_SCRIPT if clips else "") + header + video + "<h2>Events</h2>" + body, root)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs").resolve()
    episodes = collect(root)
    experiments = {}
    for episode in episodes:
        (episode["dir"] / "view.html").write_text(episode_html(root, episode))
        experiments.setdefault(episode["dir"].parent, []).append(episode)
    for directory, group in experiments.items():
        (directory / "index.html").write_text(experiment_html(root, group[0]["experiment"], group))
    (root / "index.html").write_text(index_html(root, episodes))
    print(f"{root / 'index.html'} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
