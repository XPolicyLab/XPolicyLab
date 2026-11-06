# Current production contract tests

Run the complete CPU suite with the admitted official **Python 3.11.15**.
Set `RIEMANN_EVAL_PYTHON` to that interpreter's absolute path, as described in the
[installation instructions](../README.md). Other Python versions are not a
supported substitute: the runtime identity checks include Python AST identities.

```bash
"$RIEMANN_EVAL_PYTHON" --version  # must print Python 3.11.15
"$RIEMANN_EVAL_PYTHON" -I -S -B tools/check_source_contracts.py
"$RIEMANN_EVAL_PYTHON" -B -m unittest discover -s tests -t .
```

The source-contract command uses only the standard library. The complete CPU
suite additionally requires the admitted SDK/protocol, NumPy and the configured
external prerequisites. Resolve missing prerequisites before interpreting a run.
These are manual validation commands; this policy package does not install a
GitHub Actions workflow in the upstream repository.

The suite covers all 54 task CLIs, executable graph resolution, Rule/WAM routing
and selected WAM bindings, model asset admission, public observation/action
filtering, motion sampling and action acknowledgments, empty planning chunks,
server teardown, physical GPU identity and official checkpoint/results handling.
It does not launch an Isaac scene, acquire a real compute GPU or load weights.
Passing these checks does not certify task success rates or official-cloud
acceptance.
