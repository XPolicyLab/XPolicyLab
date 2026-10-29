# Build and run

Simate beta + adaptive execution submission (default 5–50, configurable minimum). Native simulator acceptance must be checked in the target official environment. This source update does not claim a new GPU evaluation has completed.

Deploy the XPolicyLab repository directly inside the official RoboDojo workspace.
This adapter lives at `policy/SenseNova-RoboRSI/` inside that repository.
Keep the `XPolicyLab/policy/SenseNova-RoboRSI` layout: installation also installs
the framework from `../..`, and evaluation needs its shared server and utilities.
This package contains the authenticated `adaptive_2_50` source baseline. Runtime
approval limits default to 5–50; `ROBORSI_ADAPTIVE_MIN_STEPS` selects a minimum
from 2 to 50. It does not contain an installed environment.
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
conda activate RoboDojo
export ROBORSI_BOOTSTRAP_PYTHON="$(command -v python)"
cd XPolicyLab/policy/SenseNova-RoboRSI
bash install.sh
```

`bash bootstrap.sh --check` verifies the committed build inputs without downloads,
compilation or environment creation. Installation downloads public dependencies;
it uploads nothing and requires no credentials.

The bootstrap returns to `install.sh`, which installs the pinned controller
baseline (71 locked distributions plus source RoboRSI), five XPolicyLab runtime
additions, two build packages and XPolicyLab itself. It separately installs the
model tools, CLI, ICL and the selected VLA assets. The default is Simate beta;
its separate Python 3.11 virtualenv reuses the official environment's CUDA
PyTorch 2.6–2.8, and installs the other eight packages using
`simate/requirements.lock` with required SHA-256 checks. The original simulator
Python and the sibling `policy/Simate_beta` source are never modified. This path
does not download the unused pi0.5 weights or create its runtime. Public downloads need access
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
and `submission.json`, together with `simate/`: they authenticate the exact
installation inputs.

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

## Simate backend and cached assets

`policy/Simate_beta` must be present from this official XPolicyLab repository.
`simate/source.json` records its source revision and the nine inference/config
file hashes. Installation and service startup verify those files before import.
No extra source checkout, author filesystem, cluster archive or model API key is
needed.

The installer downloads `model.safetensors` from fixed Hugging Face revision
`0738aab54e0c873db76d4d2aec7a467ca1b04120` and the public tokenizer. It verifies
both complete file sizes and SHA-256 values in `simate/assets.json`; model weights
are about 13.50 GB and are not committed to Git. Complete existing files in the
checkpoint directory are checked and reused. Optional cache configuration:

```bash
# Use an existing Simate Python 3.11 environment / complete checkpoint directory.
export ROBORSI_SIM_MODEL_PYTHON=/path/to/simate-env/bin/python
export ROBORSI_SIM_CHECKPOINT=/path/to/simate-beta-robodojo-eval
# If the policy/CLI/tool/ICL environment is already installed:
bash install_simate.sh
```

With no overrides, the installer creates `runtime/simate-env` using the official
RoboDojo Python 3.11 environment and stores weights in
`checkpoints/simate-beta-robodojo-eval`. `ROBORSI_SIM_BASE_PYTHON` can select the
existing official Python 3.11 with CUDA PyTorch used to create that venv.
`ROBORSI_SIM_ASSET_CACHE` can select a directory containing the two complete
cached weight files; cache files are verified before copying. A checkpoint
selected outside the policy is linked under the standard relative checkpoint
path, so the submitted `deploy.yml` contains no machine-specific address.

Successful installation generates ignored, machine-local `simate_backend.json`
and `simate-install.json`. Those record the actual interpreter/checkpoint paths
and code/asset identities; they must be generated on the destination machine,
not copied from an author's runtime. `install.sh` finishes with the real offline
installation preflight. It does not load model weights into a GPU, start a
simulator or call an LLM API.

Start individual evaluations through this policy's standard `eval.sh` entry
described in [README.md](README.md). Concurrent scheduling is supplied as a
separate package. The adaptive controller permits 5–50 approved action steps by
default. Set `ROBORSI_ADAPTIVE_MIN_STEPS=2` to restore 2–50; the omitted approval
length is `max(5, minimum)`, while Simate proposals retain horizon 50. See the
policy README for per-instance configuration, resume and receipt requirements.
