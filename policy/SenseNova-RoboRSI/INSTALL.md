# Build and run

Extract the installation source ZIP so `XPolicyLab/` is directly inside the
official RoboDojo workspace. It preserves the full XPolicyLab source tree and
adds this policy; the PR changes only `policy/SenseNova-RoboRSI/`.
A policy-only directory requires an existing compatible
XPolicyLab checkout: installation also installs that framework from source.
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
76 hash-locked packages, and builds `openpi-client==0.1.0`
from included, authenticated sources. Build dependencies
stay in a separate build environment. The resulting tool environment must contain
exactly 77 distributions with the B-region versions. From the original 83,
the unused historical `heart-runtime==0.0.0` editable installation and the
test-only packages `pytest`, `pytest-timeout`, `iniconfig`, `pluggy` and
`Pygments` are omitted. Tests run in a separate development environment.

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

The bootstrap returns to `install.sh`, which installs the B controller
baseline (71 locked distributions plus source RoboRSI), five XPolicyLab runtime
additions, two build packages and XPolicyLab itself. It separately installs the
model tools, the independent VLA runtime, CLI, ICL and model assets. Public downloads need access
to Python.org, PyPI, GitHub, Hugging Face and Google Storage. No model key is
needed. The resulting virtualenv, caches and internal validation state remain
local under ignored runtime paths and are not part of the source package.

Once installation succeeds, use the single-case command in [README.md](README.md).
Run with the same OS user that installed the private tools, and retain their
physical directory (`runtime/tool-deployment` points to it). The official
simulator interpreter and its activation hooks must be readable/executable by
that user. Installation never rewrites the benchmark image or its permissions.

## B-region alignment and dependency cleanup

The baseline is `b-region-roborsi-runtime-environment-20261016.md` and its
GM runtime receipt. Each interpreter owns a separate lock; never combine their
requirements or install policy Torch/JAX into the benchmark environment.

| Process | Python | Runtime dependency policy |
| --- | --- | --- |
| Controller | 3.10.12 | All 72 B distributions, plus the documented XPolicyLab and build additions |
| Model-tool sandbox | 3.11.13 | 77 B versions; omit the unused editable and five test-only distributions |
| pi0.5 service | 3.11.17 | 142 B versions; original OpenPI/LeRobot/augmax/numpydantic source overlays |
| Benchmark simulator | Benchmark-owned | Use the official benchmark environment and activation scripts |

The pi0.5 service uses the fixed checkpoint and tokenizer downloaded by this
installer. Its runtime omits the unused `gcsfs`/Google Cloud/OAuth dependency
group (14 distributions). Three packages in that group require a newer protobuf
than B's `protobuf==4.25.8`; removing the unused group preserves the B inference
versions without upgrading protobuf. Arbitrary uncached `gs://` downloads and
training environments are outside this adapter's fixed inference path.

The five original `lerobot-0.1.0.dist-info` files and complete B157 package
snapshot remain under `pi05/lerobot-metadata/` as provenance. They are not
installed: their historical dependency declarations describe another LeRobot
code tree, require unused simulator/training packages, and conflict with B's
torchvision and wandb. The authenticated LeRobot source overlay remains active.
`pytest` stays in the VLA runtime because its actual model sources import it.

Installation clears inherited Python paths and active virtualenv selection,
disables user site packages, and ignores unrelated uv configuration. The VLA
check also requires `torch==2.7.1+cu126` and CUDA build `12.6`. The adapter's
separately pinned Triton assembler remains 12.4.99; the baseline document does
not establish the historical system assembler version.
