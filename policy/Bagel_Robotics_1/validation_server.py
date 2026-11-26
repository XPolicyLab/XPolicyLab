"""Official server with an optional recorder; no inference or scoring changes."""
import os
from pathlib import Path
import sys

POLICY = Path(__file__).resolve().parent
XPL = POLICY.parents[1]
sys.path[:0] = [str(XPL.parent), str(XPL)]

import setup_policy_server as official
from validation import attach


def main():
    original = official.eval_function_decorator

    def load(name, attribute):
        factory = original(name, attribute)

        def construct(config):
            model = factory(config)
            root = Path(os.environ["BAGEL_VALIDATION_TRACE"])
            attach(model, root / str(config["task_name"]) / f"server-{os.getpid()}")
            return model
        return construct

    official.eval_function_decorator = load
    official.main(official.parse_args_and_config())


if __name__ == "__main__":
    main()
