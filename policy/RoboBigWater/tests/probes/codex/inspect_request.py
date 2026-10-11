"""Summarise the first model request saved by the egress proxy: which capabilities does Codex advertise?"""
import json, re, sys
raw = open(sys.argv[1], "rb").read()
data = json.loads(raw.split(b"\n", 1)[1])
print("model", data.get("model"), "reasoning", data.get("reasoning"), "top-level tools", [t.get("name") or t.get("type") for t in data.get("tools", [])])
text = "\n".join(p.get("text", "") for it in data["input"] if isinstance(it.get("content"), list) for p in it["content"])
print("prompt chars", len(text))
for tag in ("skills_instructions", "multi_agent_role", "multi_agent_mode", "apps_instructions", "plugins_instructions"):
    print(f"  <{tag}>", "present" if f"<{tag}>" in text else "absent")
print("  tools.* names:", sorted(set(re.findall(r"tools\.([a-z_]+)", text))))
print("  skills:", re.findall(r"^- ([a-z-]+): ", text, flags=re.M))
for kw in ("web_search", "web.run", "browser", "image_gen", "spawn_agent"):
    print(f"  mentions of {kw}:", len(re.findall(kw, text)))
