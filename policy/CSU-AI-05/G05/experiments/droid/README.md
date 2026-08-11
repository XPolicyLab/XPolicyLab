# G0.5 DROID Policy Server

This is the **server side** of the DROID / Franka FR3 eval stack. It loads a G0.5
VLA checkpoint and serves actions over a **websocket + msgpack** protocol
([PROTOCOL.md](PROTOCOL.md)). The eval **client is a separate repo** and talks to
this server *only* through that protocol — no shared Python code.

```
NUC (robot control, docker)   ──zerorpc──►  Franka FR3
        ▲ zerorpc
Laptop = droid-franka-client  ──ws:8000──►  THIS server (G0.5, GPU host)
   + ZED cameras              ◄──actions───
```

---

## 1. Client — install separately

The client lives in its own repo. Clone it and follow **its** README for the
full from-scratch setup (droid submodule, `.env`, docker build of the eval env):

```bash
git clone git@github.com:OpenGalaxea/droid-franka-client.git
cd droid-franka-client
git submodule update --init --recursive     # pulls the droid fork + its submodules
cp .env.example .env                          # then edit machine-specific values
# … build the eval container & wire up NUC/robot/cameras per that README …
```

The client's `.env` points it at this server via `POLICY_HOST` / `POLICY_PORT`.
That, plus [PROTOCOL.md](PROTOCOL.md), is the **entire** contract between the two.

---

## 2. Server setup (this repo)

### Prerequisites
- This repo installed with its `.venv` (uv). Run all commands from the **repo root**.
- A CUDA GPU (~12 GB free for the G0.5 2B checkpoint).
- A G0.5 DROID checkpoint directory containing `.hydra/config.yaml`,
  `checkpoints/model_state_dict.pt`, and `dataset_stats.json`.
- Shared resources downloaded at the repository root:
  `checkpoints/action_tokenizer.pt` and
  `checkpoints/qwen3_5_2b_base_processor/`.

### Start the server

```bash
source .venv/bin/activate
CHECKPOINT_DIR=checkpoints/g05-droid \
POLICY_PORT=8000 \
POLICY_DEVICE=cuda:0 \
bash experiments/droid/start_server.sh \
    model.model_arch.discrete_action=true model.model_arch.continuous_action=false
```

- `CHECKPOINT_DIR` (required) — the checkpoint dir.
- `POLICY_PORT` (default `8000`), `POLICY_DEVICE` (default `cuda`).
- Trailing args are passed through as Hydra overrides. The `discrete_action=true`
  pair above is for the CoT/discrete G0.5 checkpoint; drop it for a continuous one.
- The script sets `PYTHONPATH=src` (g05 is a src-layout package, not pip-installed)
  and launches `scripts/serve_policy.py` with `eval_embodiment=Droid_Franka`.

The server is ready once it logs the policy-server banner and binds `0.0.0.0:PORT`.
Point the client's `POLICY_HOST`/`POLICY_PORT` at it and run the client's eval loop.

---

## 3. Notes

- **Interface.** [PROTOCOL.md](PROTOCOL.md) is the single coupling point; keep it in
  sync with the copy in the client repo.
- **Validation:** end-to-end verification should start the server, then confirm
  a test client connects, sends one obs, and gets back a valid `action` (`right_arm[7]` +
  `right_gripper[1]`) with `cot_text` (bbox + subtask) and `need_obs`. No missing keys.
