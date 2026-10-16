"""Private adapter process; run with the benchmark owner's Python interpreter."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policylab.api import InvalidAction, validate_transition


def main():
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # Native library stdout stays out of the JSON protocol.
    cfg = json.loads(Path(sys.argv[1]).read_text())
    filename, name = cfg["adapter"].rsplit(":", 1)
    sys.path.insert(0, str(Path(filename).parent))
    adapter = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "initialize":
                    spec = importlib.util.spec_from_file_location("benchmark_adapter", filename)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    adapter = getattr(module, name)(Path(cfg["environment_dir"]), cfg["options"])
                    result = {"initialized": True}
                elif op == "close":
                    if adapter is not None:
                        adapter.close()
                        adapter = None
                    channel.write(json.dumps({"closed": True}) + "\n")
                    break
                elif op == "reset":
                    result = validate_transition(adapter.reset(request["episode"], Path(request["record_dir"]), Path(request["observation_dir"])))
                elif op == "vla_propose":
                    result = adapter.vla_propose(request["subgoal"])
                elif op == "step":
                    result = validate_transition(adapter.step(request["action"]))
                elif op == "finish":
                    result = validate_transition(adapter.finish())
                    if not result["done"]:
                        raise ValueError("finish() must return done=True")
                else:
                    raise ValueError("Unknown private operation")
                channel.write(json.dumps(result, allow_nan=False) + "\n")
            except InvalidAction as exc:
                channel.write(json.dumps({"invalid_action": str(exc)}) + "\n")
            except Exception:
                traceback.print_exc()
                channel.write(json.dumps({"error": "Adapter failed; see simulator.log"}) + "\n")
                break
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
