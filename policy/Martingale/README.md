# Martingale

**Contributor:** Martingale | **Paper:** Martingale: A Self-Improving Harness for Robot Agents that Learn from Experience | **arXiv:** Unavailable | **Original code:** [bundled Martingale runtime](runtime/)

Martingale is a code-as-policy agent: one fixed language model controls the robot through persistent Python cells, a shared perception/motion API and one skill library. It uses RGB, camera calibration and proprioception, and sends simulator joint actions. This adapter supports evaluation and opt-in self-play practice with `bench_name=RoboDojo`, `env_cfg_type=arx_x5` and `action_type=joint`, one environment per process.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Run from `XPolicyLab/policy/Martingale` in the RoboDojo workspace. The adapter includes the inference source, skill library and default configuration. Use a Linux x86-64 policy host with a CUDA 12.8-compatible driver, a GPU and the system `bubblewrap` executable. On Ubuntu, install the latter with `sudo apt-get install bubblewrap`. The policy host must permit unprivileged user namespaces for the cell worker. Node.js and npm are needed on the machine that runs Codex. For hosted evaluation, inject `MARTINGALE_CODEX_TOKEN` before following the installation steps.

```bash
conda create -n martingale python=3.12 -y
conda activate martingale
bash install.sh
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c runtime/constraints-tested.txt setuptools==80.9.0 einops psutil pycocotools \
  'sam3 @ git+https://github.com/facebookresearch/sam3.git@31cb7e624e4fe95301b1be1d4e809111593d2e45'
# Local provider only. Hosted evaluation uses the injected evaluator token.
if [[ -z ${MARTINGALE_CODEX_TOKEN:-} ]]; then
  npm install -g @openai/codex@0.171.0-alpha.13
  export CODEX_HOME="${XDG_STATE_HOME:-$HOME/.local/state}/martingale-codex"
  install -d -m 700 "$CODEX_HOME"
  codex -c 'cli_auth_credentials_store="file"' login
fi
hf auth login
bash download_checkpoint.sh
martingale preflight --isolated
python -c 'import torch; from sam3.model_builder import build_sam3_image_model; assert torch.cuda.is_available(), "CUDA is required"'
```

The checkpoint script downloads the official `sam3.pt` at a pinned revision and verifies its SHA-256. Grant the Hugging Face account access before downloading. The Codex account requires access to the model selected in [inference.json](inference.json). [SAM3 checkpoint access](https://huggingface.co/facebook/sam3)

### Hosted Codex on Vultr

The submitter's Codex service listens on `127.0.0.1:4500` on the VM.
The public health URL is `https://codex.martingale-robotics.com/readyz`.
The evaluator URL is `wss://codex.martingale-robotics.com/episode`. For hosted
evaluation, inject the submitter's `MARTINGALE_CODEX_TOKEN` secret into the policy
host and use `eval.sh` normally. The steps below provision the submitter's VM.

#### 1. Verify DNS and the existing VM service

The `codex` A record in the `martingale-robotics.com` DNS zone should point to
`96.30.196.32`. A 30-minute TTL is fine. From your Mac, check the address and
log in to the VM:

```bash
dig +short codex.martingale-robotics.com A
ssh root@96.30.196.32
```

In the VM shell, check the existing Codex service. Expect `active` and `HTTP 200`.

```bash
systemctl is-active martingale-codex
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:4500/readyz
```

#### 2. Configure public HTTPS

In Vultr, open **Compute → Instances → martingale-codex → Settings → Firewall**.
If a group is attached, allow inbound TCP ports 80 and 443 in that group and
keep port 22 available for administration. If the dropdown is empty or shows
no group, continue directly to the Ubuntu firewall check. [Vultr instance firewall settings](https://docs.vultr.com/products/compute/instances/cloud-compute/networking/enable-firewall)

On the VM, run `sudo ufw status`. If UFW is active, run
`sudo ufw allow 80/tcp` and `sudo ufw allow 443/tcp`. Both application
processes bind to loopback. RoboDojo only needs outbound TCP 443.

On the VM, run `sudo apt-get update && sudo apt-get install -y caddy`.
On this dedicated VM, use the following site block in `/etc/caddy/Caddyfile`.
Preserve any existing sites if the VM is shared. [Caddy installation](https://caddyserver.com/docs/install)

```caddyfile
codex.martingale-robotics.com {
    @ready {
        method GET
        path /readyz
        header !Upgrade
    }
    handle @ready {
        reverse_proxy 127.0.0.1:4500
    }
    handle /episode {
        reverse_proxy 127.0.0.1:4510
    }
    handle {
        respond 404
    }
}
```

Validate and start Caddy on the VM:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

From a second Mac terminal, check the public route without an SSH tunnel:

```bash
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' https://codex.martingale-robotics.com/readyz
```

Expect `HTTP 200`. The site's root returns `404` by design. If the public check
fails, check `sudo ufw status`, the attached Vultr firewall group and
`sudo journalctl -u caddy -n 50 --no-pager` on the VM. [Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https)

#### 3. Create the evaluator credential

Generate the single-line evaluator token once on the VM. Keep the existing
Codex app-server credential in `/var/lib/martingale/endpoint.token` private
to the VM. Deliver only `robodojo.token` through the evaluator's secret store.

```bash
sudo -u martingale sh -c '
  umask 077
  test -s /var/lib/martingale/robodojo.token ||
    openssl rand -hex 32 > /var/lib/martingale/robodojo.token
'
```

#### 4. Deploy and test the episode service

From the submission checkout on a computer with administrator SSH access to
the VM, run the installer. It copies the CPU service and the pinned inference
configuration, installs the WebSocket dependency in a private virtual
environment, and starts `martingale-episode.service` on loopback port 4510.
It creates the evaluator token if needed. The final test goes through public
HTTPS and requires Codex to report a caller-generated verification word and
the color of an image returned by the Python tool.

```bash
bash policy/Martingale/install_episode_service.sh --bundle | ssh root@96.30.196.32 'bash -s'
```

With the checkout on a different machine, run the command before the pipe
and redirect its output to `install-martingale-episode.sh`. Download that file
to your Mac and run `ssh root@96.30.196.32 'bash -s' < ~/Downloads/install-martingale-episode.sh`.
A successful installation prints `PASS: authenticated episode, live Codex turn, Python callback, text and camera-image response`.
Service diagnostics are available with `journalctl -u martingale-episode -n 40 --no-pager`.

#### 5. Run the submitted policy

Supply the single-line value of the VM's `/var/lib/martingale/robodojo.token`
privately to the runner as `MARTINGALE_CODEX_TOKEN`. Run the installation and
`eval.sh` commands from this README. The policy automatically opens the public
endpoint for ordinary evaluation when that variable is present. The evaluator
needs outbound TCP 443. SAM3, the isolated Python worker, skill library,
simulator and robot actions stay on the evaluation host. Each episode owns its
Codex thread and its tool callback connection. A completed cell retains its
request ID when a response is lost. The VM refuses a continuation if it cannot
confirm the state of the previous turn.

For local Codex, leave the evaluator token unset and sign in on the policy
host as described below. Self-play and geometry fitting use the local provider.
Verify a real simulator episode before recording a RoboDojo benchmark result.

<details>
<summary>Submitter setup on a fresh Ubuntu VM</summary>

Run the following on the VM with an administrator account. The service uses
the model and reasoning effort specified in the submission.

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm openssl
sudo npm install --global --prefix /usr/local @openai/codex@0.171.0-alpha.13
id martingale >/dev/null 2>&1 || sudo useradd --system --user-group --create-home \
  --home-dir /var/lib/martingale --shell /usr/sbin/nologin martingale
sudo install -d -m 700 -o martingale -g martingale \
  /var/lib/martingale/.codex /var/lib/martingale/workspace
sudo -Hu martingale /usr/local/bin/codex -c 'cli_auth_credentials_store="file"' login --device-auth
sudo -Hu martingale /usr/local/bin/codex -c 'cli_auth_credentials_store="file"' login status
sudo -u martingale sh -c '
  umask 077
  test -s /var/lib/martingale/endpoint.token ||
    openssl rand -hex 32 > /var/lib/martingale/endpoint.token
'
```

Complete the device login in your browser. The episode service uses
`/var/lib/martingale/endpoint.token` on the VM. Install the Codex service and
check that it is running.

```bash
sudo tee /etc/systemd/system/martingale-codex.service >/dev/null <<'EOF'
[Unit]
Description=Martingale Codex endpoint
After=network-online.target
Wants=network-online.target

[Service]
User=martingale
Group=martingale
Environment=HOME=/var/lib/martingale
Environment=CODEX_HOME=/var/lib/martingale/.codex
WorkingDirectory=/var/lib/martingale/workspace
UMask=0077
ExecStart=/usr/local/bin/codex app-server \
  -c model=gpt-6.2-astra -c model_reasoning_effort=xhigh \
  -c cli_auth_credentials_store=file -c sandbox_mode=read-only -c approval_policy=never \
  --listen ws://127.0.0.1:4500 --ws-auth capability-token \
  --ws-token-file /var/lib/martingale/endpoint.token
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now martingale-codex
sudo systemctl status martingale-codex --no-pager
```

</details>


## Data Processing

Unsupported: the agent is not trained on trajectory data. `process_data.sh` is omitted (evaluation-only adapter).

## Training

Unsupported: no weights are trained and there is no checkpoint directory. `train.sh` is omitted (evaluation-only adapter). The skill library, procedures and lessons in the inference package are code and text prepared before evaluation, in part from practice episodes on the benchmark's tasks. The learned model components are the hosted language model and the public SAM3 checkpoint. `ckpt_name` is a label only (use `martingale`).

## Evaluation

```bash
cd XPolicyLab/policy/Martingale
export CODEX_HOME="${XDG_STATE_HOME:-$HOME/.local/state}/martingale-codex"
export MARTINGALE_OUTPUT_ROOT="${XDG_STATE_HOME:-$HOME/.local/state}/martingale"
mkdir -p -m 700 "$MARTINGALE_OUTPUT_ROOT"
# Use the Conda environment with the RoboDojo client on this machine.
export ROBODOJO_ENV=robodojo
conda run -n "$ROBODOJO_ENV" python -m pip install -e ../..
bash eval.sh RoboDojo stack_bowls martingale arx_x5 joint 0 0 0 martingale "$ROBODOJO_ENV"
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

For practice, append `--self-play` to the command above. On a split deployment,
set `MARTINGALE_SELF_PLAY=1` on the policy host. This enables bounded, text-only
`robo.llm_query(context, question)` subcalls to the same configured model and
`robo.skill_stage(...)` for lesson and Python skill updates. Subcalls have no
robot controls or further subcalls and are recorded under `workspace/rlm/`.
Staged edits are published after a normal episode verdict and worker shutdown;
before/after contents and evidence are saved under `self_play_updates/`.
Use a writable practice library checkout; subsequent runs of that checkout
read the published updates. Default evaluation exposes neither API and performs
no publication. Weights are unchanged. Do not combine this flag with
geometry-fitting mode.

## Configuration

| Variable / key | Meaning |
|---|---|
| `MARTINGALE_INFERENCE_CONFIG` | Optional absolute path. The launcher defaults to the bundled `inference.json` for every task. |
| `MARTINGALE_OUTPUT_ROOT` | Private directory for per-episode records (prompts, cells, provider events). |
| `MARTINGALE_SELF_PLAY` | `1` enables practice-only LM subcalls; unset or `0` keeps ordinary evaluation. |
| `SAM3_CHECKPOINT_PATH` | Defaults to `${XDG_CACHE_HOME:-$HOME/.cache}/martingale/sam3.pt`, the location used by the download script. |
| `MARTINGALE_CODEX_TOKEN` | Private bearer token enabling the hosted endpoint for ordinary evaluation. |
| `MARTINGALE_ACCOUNT_REGISTRY` | Private operator registry for locally authenticated Codex episodes. |
| `CODEX_HOME`, `MARTINGALE_CODEX_HOME` | Explicit credential store when no registry is configured. |
| `MARTINGALE_XPOLICYLAB_ROOT` | Explicit XPolicyLab checkout for the split-machine scripts; defaults to the installed adapter layout. |
| `deploy.yml: poll_s` | Seconds between polls while the model is thinking. |
| `deploy.yml: inference_config`, `result_dir` | Optional explicit paths; the environment variables above are used when they are null. |
| Accepted `deploy.yml` keys | `policy_name`, `protocol`, `host`, `port`, `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type`, `seed`, `action_type`, `gpu_id`, `eval_batch`, `poll_s`, `inference_config`, `result_dir`, and the server keys XPolicyLab adds (`relay_url`, `ws_ping_interval_s`, `ws_ping_timeout_s`). Any other key is rejected at start-up. |

## Notes

- **Task text.** The agent's task text is the official Detail Description, `obs["extra_info"]["detailed_instruction"]`, stated unchanged and followed by the current `obs["instruction"]`. When the observation does not carry the field, `obs["instruction"]` alone is stated. The episode's `status.json` records which source was used.
- **Single fixed system.** The same model, prompt framework, tool API and skill library serve every task. The adapter rejects any `deploy.yml` key it does not accept, and nothing in the runtime branches on task name, layout or seed. The library contains task-specific procedures, lessons and routines; which of them applies is decided by the agent from the task text.
- **`deploy.py` differs from `demo_policy`.** The model thinks for minutes between actions, so the policy answers WAIT while it reasons and `deploy.py` polls again without stepping the simulator. It drives one environment per process (`eval_batch: false`; the batch entry points accept exactly one environment). Reads that change nothing (the observation, the episode-end check, the verdict) are retried automatically up to three times, 1.5 s apart; any other failing call, or a read whose retries are used up, raises to the evaluation client as in `demo_policy`. There is no operator entry point. A new `reset` ends any earlier session that never received its episode's end; every episode starts with a fresh session and memory. The adapter adds the step count to each observation and sends no task name, seed or layout identifier.
- **Episode end.** There is no finish tool. The simulator supplies the terminal success or failure verdict. The XPolicyLab debug client supplies a step limit. Completing it records `debug_completed` in the session status.
- **Interruptions.** A model-provider failure pauses the conversation on the policy host; see the package README for resuming it. Faults between the evaluation client and the policy server are handled as described above.
