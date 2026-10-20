# Build and run

RC1 adapter: real evaluation acceptance for one shared VLA and ten concurrent cases per GPU is pending. This PR does not claim that concurrency validation has passed.

Deploy the XPolicyLab repository directly inside the official RoboDojo workspace.
This adapter lives at `policy/SenseNova-RoboRSI/` inside that repository.
Keep the `XPolicyLab/policy/SenseNova-RoboRSI` layout: installation also installs
the framework from `../..`, and evaluation needs its shared server and utilities.
This package contains the `adaptive_2_50` source, allowing 2–50 approved steps
with default 5; it does not contain an installed environment.
Keep the official simulator image/Conda environment and Assets unchanged.

## Prerequisites

Linux x86_64, Bash, `setsid`, a seed Python with tar extraction filters (the
simulator's Python 3.11.17 works), and an NVIDIA CUDA-12-compatible driver.
For building the policy's Python on Ubuntu/Debian, install system build tools
if they are absent:

```bash
sudo apt-get install build-essential pkg-config libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev uuid-dev
```

The bootstrap builds official CPython 3.10.12, not a copy of an author's runtime.
It checks real memfd/CLOEXEC/read/write and required standard libraries before
using that interpreter. `ROBORSI_PYTHON` can instead select an already compatible
3.10.12. `ROBORSI_BOOTSTRAP_PYTHON` selects the seed Python;
`ROBORSI_BOOTSTRAP_ROOT` selects build/cache storage (default `runtime/bootstrap`).
uv 0.8.9 and CPython source have fixed SHA-256 values in `bootstrap.py`.
The installer does not modify system Python or install system packages.

## Independent model tools

The installer downloads the fixed Python 3.11.13 standalone release, installs
81 hash-locked packages, and builds `openpi-client==0.1.0`
from included, authenticated sources. Build dependencies
stay in a separate build environment. The resulting tool environment must contain
exactly 82 distributions with the pinned versions. The unused historical
`heart-runtime==0.0.0` editable installation is omitted from the original 83.

`tools/runtime.json` pins the interpreter, package locks and source identities.
The adapter retains its original tool scripts, OpenPI client and sandbox
constructor. No legacy framework or placeholder package is installed.

Installation verifies package/source payloads and imports the actual tools,
NumPy, SciPy and Pillow inside the original sandbox. It requires no
`original-model-environment.tar.gz`, external author checkout or model key.

## One command

```bash
cd XPolicyLab/policy/SenseNova-RoboRSI
bash install.sh
```

`bash bootstrap.sh --check` verifies the committed build inputs without downloads,
compilation or environment creation. Installation downloads public dependencies;
it uploads nothing and requires no credentials.

The bootstrap returns to `install.sh`, which installs the pinned controller
baseline (71 locked distributions plus source RoboRSI), five XPolicyLab runtime
additions, two build packages and XPolicyLab itself. It separately installs the
model tools, VLA157, CLI, ICL and model assets. Public downloads need access
to Python.org, PyPI, GitHub, Hugging Face and Google Storage. No model key is
needed. The resulting virtualenv, caches and internal validation state remain
local under ignored runtime paths and are not part of the source package.

Once installation succeeds, configure `ROBORSI_CODEX_BASE_URL`,
`ROBORSI_CODEX_MODEL` and `CODEX_API_KEY` in your runtime environment, then use
the single-case command in [README.md](README.md). The service URL and model
have no defaults; evaluation fails early if either is missing or empty.
Installation does not require these runtime values.
Run with the same OS user that installed the private tools, and retain their
physical directory (`runtime/tool-deployment` points to it). The official
simulator interpreter and its activation hooks must be readable/executable by
that user. Installation never rewrites the benchmark image or its permissions.

## Installation and validation boundaries

Run `bash install.sh` at the final physical policy path. This entry performs the
legacy source installation directly; it is not the separate benchmark-image
`--plan` / `--install` dispatcher. Installed interpreters, private tool symlinks,
checkpoints and receipts must be generated or verified at the new location. Do
not copy an old receipt or rewrite absolute paths to claim installation success.
Keep the source locks and manifests under `vendor/`, `tools/`, `pi05/`, `icl/`
and `submission.json`: they authenticate the exact installation inputs.

The official workspace, complete native Assets, simulator Python 3.11.17 and
activation hooks, and matching Kit extension bundle remain external
prerequisites. `install.sh` prepares the policy, model-tool and VLA environments;
it does not assemble those native benchmark inputs. Finish installation and
offline preflight before sharing the code and weights between evaluations.

The optional `environment/run.py` namespace launcher is included in this policy
directory, but this PR supplies no deployment plan. Pass a current verified plan
explicitly to `environment/run.py --plan /absolute/current-plan.json` when using
that launcher. Direct installation and evaluation use the official workspace
layout described above and generate their own local installation state.

Shared VLA evaluation is opt-in through `ROBORSI_SHARED_VLA_PORT`; each case
otherwise starts its own VLA. Acceptance for one shared VLA and ten concurrent
cases per GPU remains pending. This PR includes no ten-case orchestration harness
or concurrency validation results. Source installation, native simulator startup,
shared-service concurrency and full task success require their respective checks
in the target workspace.
