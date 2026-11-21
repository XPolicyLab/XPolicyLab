# Installation

Run the following commands from `XPolicyLab/policy/JoyAI_VL` unless stated otherwise.
Example paths must be replaced with paths on your machine.

## Runtime components

| Component | Requirement |
| --- | --- |
| Host | Linux, NVIDIA GPU / driver, access to the Docker daemon |
| Environment and policy server | Working RoboDojo checkout, its assets, Isaac Sim 5.1 and cuRobo in the Isaac Python environment |
| Agent | Lightweight Docker image built by the policy installer; Python 3.11, NumPy 1.26.0, Pillow 11.3.0 |
| Segmentation | Separate CUDA-enabled SAM3 environment and SAM3 checkpoint |
| Model | Chat Completions endpoint supporting images, tool calls, reasoning history, and the configured chat-template fields |
| Setup helpers | Python 3.11 or newer |

Prepare RoboDojo, its assets and Isaac/cuRobo using the
[RoboDojo project documentation](https://robodojo-benchmark.com/).
The policy server uses this environment on the host or in a simulator container.
Docker runs the agent and its API proxy.

## 1. Open the policy directory

Use this checkout as `RoboDojo/XPolicyLab`, then enter `XPolicyLab/policy/JoyAI_VL`.
Run the commands below from that policy directory.

## 2. Configure the model

Request an API key from either [qizhangyang.1@jd.com](mailto:qizhangyang.1@jd.com)
or [tiankanghui.1@jd.com](mailto:tiankanghui.1@jd.com).
Create a private key file outside the repository, for example
`$HOME/.config/joyai-vl/api-key`, with permissions `0600`. Use your endpoint's supplied
API key; do not place it in command-line arguments, configuration JSON, or Git.

```bash
python3 tools/configure.py \
  --robodojo-repo /path/to/RoboDojo \
  --isaac-python /path/to/isaac-env/bin/python \
  --api-base https://modelservice.jdcloud.com/v1 \
  --model joyai_vl_testing \
  --key-file "$HOME/.config/joyai-vl/api-key" \
  --sam3-url http://127.0.0.1:18300
```

`JoyAI-VL` is the public model name; `joyai_vl_testing` is the serving model ID.
The base URL ends before `/chat/completions`; both the URL and model ID remain configurable.
The key file is read by the egress proxy and is not mounted in the model's agent container.
The proxy adds `Authorization: Bearer ...` and a fresh `Trace-Id` to each model request.
See the [API request example](../README.md#model-api-access).
For a server without authentication, provide a separate empty key file.

The generated `config.env` is private, mode `0600`, and is never overwritten automatically.
The egress container must be able to reach the API URL: `127.0.0.1` inside that container
refers to the container itself. Use a reachable host address for a model served on the host.
The policy-side SAM3 URL is resolved on the policy server, where localhost is appropriate.

Optional `--sim-cache /path/to/cache` preserves Isaac caches. A private
`--tmp-dir /path/to/temp` avoids permissions conflicts on shared hosts.
For additional simulator library directories, edit `EXTRA_LD_LIBRARY_PATH` in the installed
policy's `config.env`.

## 3. Build the agent and install the policy dependency

```bash
bash /path/to/RoboDojo/XPolicyLab/policy/JoyAI_VL/install.sh
```

This builds the local image `joyai-vl/agent:preview` and installs `websockets>=13` into
the policy's `.cache/policy-deps`, ahead of Isaac's bundled version on `PYTHONPATH`.
The reference deployment used policy-side websockets 17.4; see
[runtime references](../metadata/runtime_reference.json). The install script resolves
an available version satisfying the API requirement, so record the version it prints.

## 4. Start the shared SAM3 service

Install [SAM3](https://github.com/facebookresearch/sam3) and CUDA-enabled PyTorch in a
separate environment following its installation instructions. The reference checkout
and observed dependency versions are recorded in `metadata/runtime_reference.json`.
For a fresh service environment, follow SAM3's current Python 3.12+ and PyTorch 2.7+
requirements. The evaluated environment used Python 3.11 and is recorded in the runtime
reference. Obtain the **SAM3** checkpoint under its own access terms.

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/sam3-env/bin/python services/sam3_server.py \
  --checkpoint /path/to/sam3.pt \
  --port 18300 --cache-images 24
```

Run this in a separate terminal, and check readiness:

```bash
curl --fail http://127.0.0.1:18300/health
sha256sum /path/to/sam3.pt
```

Reference checkpoint SHA-256:
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

Use this policy's `services/sam3_server.py` in the command above. That exact
service was used in the evaluated deployment. The policy also retains an upstream
vendored service for source integrity; it is a different implementation. Multiple
evaluation processes can share one local service. GPU memory planning must include
both Isaac/cuRobo and SAM3 when they share a GPU.

Obtain model access, SAM3 checkpoints and benchmark assets separately, then configure
the paths and API settings for your deployment.
