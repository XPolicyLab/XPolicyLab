# go3_agent_preview smoke test

Four tasks × ten layouts on one machine, before spending hours on the full official
sweep. It answers "can this host run an evaluation of this adapter at all?" — not
"how good is the policy?".

```bash
cd <robodojo_root>/XPolicyLab/policy/go3_agent_preview/tests/smoke

# code branch only: no backend, no checkpoint
./smoke_test.sh --robodojo-root <robodojo_root> \
  --policy-env <robodojo_root>/XPolicyLab/.venv-geact

# both branches: start the WAM backend this run
WAM_LAUNCHER=../../wam/run_geact_backend.sh \
CKPT=/path/to/preview_model/checkpoints \
STAT_PATH=/path/to/preview_model/stats.json \
WAM_VAE_DIR=/path/to/preview_model/vae \
WAM_QWEN_DIR=/path/to/Qwen3.5-2B \
WAM_LATENT_STATS_PATH=/path/to/preview_model/vae/latent_stats.pth \
  ./smoke_test.sh --robodojo-root <robodojo_root> \
    --policy-env <robodojo_root>/XPolicyLab/.venv-geact --wam-gpu 1
```

Exit codes: `0` pass, `1` preflight, `2` a run did not finish, `3` a check failed,
`4` a task switched to WAM but the backend never answered.

## What it checks

Every episode starts in the code branch; WAM runs only after a task hits one of its
declared abandonment checkpoints. So a run is judged in three independent layers:

| Layer | Source | What a failure means |
| --- | --- | --- |
| run | `eval_result/.../<run_id>/_result.json` | `eval_time` drifted from the requested count, duplicate or short layout sets, fatal log lines |
| code | `events.jsonl`: `episode_start`, `code_tick` | the env never reached the adapter, or the program never ran |
| wam | `events.jsonl`: `switch_boundary` → `wam_chunk` → `first_wam_action` | a switch occurred and the backend did not serve it |

A task that never switches is reported as **"WAM not exercised"**, never as a WAM
pass. On a healthy host most of these four tasks finish in code alone, so treat that
note as normal — the backend layer only proves itself when a switch happens. To
force one, run a task without a code plan, such as `sweep_blocks`, which switches on
its first action, or a task whose code plan reliably exhausts (see `SWITCH_SOURCES`
in `gm/runtime/session.py`), or raise `--request-timeout` so the WAM path has room.

The checker also flags `wam_error`, `code_worker_startup_timeout` and PhysX refills
as notes or failures rather than hiding them in a log.

## Default tasks

`stack_blocks`, `push_T`, `deposit_coin`, `play_Xylophone`. All four have short
code-side plans and no PhysX abandon/refill.

Deliberately excluded: `general_pickup`. Its code planning runs for minutes per
request, so under the client's default 120 s budget it fails on any host — that would
test the clock, not the machine.

## Nothing is host-specific

Every path comes from a flag or the environment. There is no assumed checkout path,
no conda env name, no GPU index and no asset location in the script. `--env-script`
sources a file before each eval when a host needs `module load`, `LD_LIBRARY_PATH`
or Isaac paths.

Layouts are pinned through RoboDojo's resume manifest, the same mechanism
`scripts/run_robodojo_layout_range.sh` uses: the wanted ids stay in the queue and the
rest are marked abandoned, so both `--layouts 0-9` and a scattered set such as
`0-4,25-29` evaluate exactly those layouts.

## Artifacts

Under `<robodojo>/outputs/go3_smoke_<stamp>/` (or `--out`):

- `logs/<task>.log` — the full eval log
- `events/<task>/events.jsonl` — the adapter's own record of code ticks and switches
- `configs/deploy-<task>.yml` — the per-task deploy config actually used
- `manifests/<task>.log` — which layouts were pinned and how many were carried over
- `verify.json` — the checker's full verdict
