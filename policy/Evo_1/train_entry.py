"""Set the requested seed and execute the upstream trainer without editing it."""
import argparse
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evo1-source", required=True)
    parser.add_argument("--evo1-seed", required=True, type=int)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Upstream imports a namespace package named "model". Do not let this
    # adapter's model.py shadow it via the launcher script's sys.path entry.
    adapter_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != adapter_dir]
    from accelerate.utils import set_seed
    set_seed(args.evo1_seed)
    script = Path(args.evo1_source).resolve() / "Evo_1/scripts/train.py"
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(script.parent.parent))
    sys.argv = [str(script)] + (args.args[1:] if args.args[:1] == ["--"] else args.args)
    runpy.run_path(str(script), run_name="__main__")
