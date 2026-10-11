#!/usr/bin/env python3
"""After an episode: what did the agent do besides running robo? Reads the Codex session record.

Codex runs this model in code mode: every tool call is a small JavaScript
script. A script may only call tools, or it may also compute something
(camera projection, arithmetic, loops). Both are recorded here, as are shell
commands other than robo, python use, sub-agents and web search.
"""
import glob, json, re, sys
from collections import Counter

run = sys.argv[1]
calls, scripts, shell, usage = Counter(), [], [], []
for path in glob.glob(f"{run}/codex_home/sessions/*/*/*/*.jsonl"):
    for line in open(path):
        try:
            payload = json.loads(line).get("payload") or {}
        except ValueError:
            continue
        kind = payload.get("type")
        if kind in ("function_call", "custom_tool_call"):
            text = str(payload.get("arguments") or payload.get("input") or "")
            name = payload.get("name")
            if name != "exec":
                calls[f"direct:{name}"] += 1
            else:
                scripts.append(text)
            for tool in re.findall(r"tools\.([a-zA-Z_]+)", text):
                calls[tool] += 1
            shell += re.findall(r'cmd:\s*"((?:[^"\\]|\\.)*)"', text)
        elif kind == "token_count":
            last = (payload.get("info") or {}).get("last_token_usage")
            if last:
                usage.append(last)

CALL = re.compile(r"(?:await\s+)?tools\.[a-zA-Z_]+\(\{.*?\}\)", re.S)
COMPUTE = re.compile(r"Math\.|\bfor\s*\(|\bwhile\s*\(|\.map\(|\.reduce\(|=>|\[\s*-?\d|[-+*/]\s*-?\d|\d\s*[-+*/]|\*\*")


def computes(script):
    """True if arithmetic, loops or Math remain once the tool calls and string literals are removed."""
    rest = CALL.sub("", script)
    rest = re.sub(r'"(?:[^"\\]|\\.)*"', '""', rest)  # string literals do not count
    return bool(COMPUTE.search(rest))


computing = [s for s in scripts if computes(s)]
robo = [c for c in shell if re.match(r"\s*robo\b", c)]
other = [c for c in shell if not re.match(r"\s*robo\b", c)]
report = {
    "tool_calls": dict(calls),
    "scripts": len(scripts),
    "scripts_with_computation": len(computing),
    "computation_examples": [re.sub(r"\s+", " ", s)[:240] for s in computing[:5]],
    "robo_shell_calls": len(robo),
    "other_shell_calls": len(other),
    "other_shell_examples": [c[:120] for c in other[:8]],
    "used_python": any(re.search(r"\bpython3?\b", c) for c in shell),
    "used_code_for_computation": bool(computing) or any(re.search(r"\bpython3?\b", c) for c in shell),
    "spawned_agents": sum(v for k, v in calls.items() if "spawn" in k),
    "web_search": sum(v for k, v in calls.items() if "web" in k or "search" in k),
    "model_requests": len(usage),
    "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
    "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in usage),
    "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
    "last_context_tokens": usage[-1].get("input_tokens") if usage else None,
}
json.dump(report, open(f"{run}/audit.json", "w"), indent=1)
print(json.dumps(report, indent=1))
