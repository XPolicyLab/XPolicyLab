"""Run inside the agent container: what can be reached or read?"""
import json, os, socket, urllib.request

def http(url, timeout=6):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as r:
            return f"HTTP {r.status}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:80]}"

def tcp(host, port, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return "connected"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:80]}"

gw = os.environ.get("HOST_GATEWAY", "")
model_host = os.environ.get("MODEL_HOST", "")
model_path = os.environ.get("MODEL_BASE_PATH", "/v1")
robo_port = int(os.environ.get("HOST_ROBO_PORT", "28700"))
checks = {
    "robo agent route via egress (want HTTP 200)": http("http://egress:18700/v1/obs/state.json"),
    "robo admin route via egress (want 403)": http("http://egress:18700/admin/result"),
    "model route via egress (want HTTP 200)": http("http://egress:8080" + model_path + "/models", 60),
    "non-model path on gateway via egress (want 403)": http("http://egress:8080/"),
    "model endpoint directly (want failure)": tcp(model_host, 443) if model_host else "skipped",
    "public internet 1.1.1.1:443 (want failure)": tcp("1.1.1.1", 443),
    "public DNS name (want failure)": tcp("pypi.org", 443),
    "host admin port directly (want failure)": tcp(gw, robo_port + 1) if gw else "skipped",
    "host agent port directly (want failure)": tcp(gw, robo_port) if gw else "skipped",
}
paths = os.environ.get("HOST_PATHS", "").split(":") + ["/run/model-key", "/codex-home/auth.json", "/work", "/codex-home/config.toml"]
checks["paths present"] = {p: os.path.exists(p) for p in paths}
checks["env with KEY/TOKEN"] = [k for k in os.environ if "KEY" in k or "TOKEN" in k or "SECRET" in k]
print(json.dumps(checks, indent=1, ensure_ascii=False))
