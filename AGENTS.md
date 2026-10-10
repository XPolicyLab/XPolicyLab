# XPolicyLab Agent Guide

XPolicyLab wraps each robot policy as a self-contained adapter under `policy/<POLICY>/`. The policy
server imports it as `XPolicyLab.policy.<POLICY>.model`, so this checkout is always used as a
package inside a parent workspace — never as the top-level project.

- Submission standard: [CONTRIBUTING.md](CONTRIBUTING.md). Reference adapter: `policy/demo_policy/`.
  Data formats: [README](README.md#-standard-data-formats).
- Two skills carry the end-to-end workflows — `xpolicylab-model-integration` (build an adapter) and
  `xpolicylab-adapter-check` (audit one before a PR). They live in `.agents/skills/`, which
  `.cursor/skills` and `.claude/skills` symlink to, so Cursor, Claude Code and Codex all load them.
- Root docs stay bilingual: when you edit `README.md`, update `README_zh.md` in the same change (and
  the reverse). Keep structure, anchors, code blocks, and tables aligned; translate prose only.
  English remains the source of truth on conflict.

The rules below apply to every change in this repo. Their rationale is in CONTRIBUTING.md.

## Images are RGB from end to end

`decode_image_bit` and `decode_obs_images` return RGB, and the policy server hands `update_obs` /
`update_obs_batch` RGB. Treat this as settled, and note that it holds for a reason no caller can
reproduce: stored image bits come in **two byte formats**, and `decode_image_bit` reads a marker
inside each buffer to tell them apart and swap only where a swap is owed.

- **legacy** — a JPEG written by handing an RGB array straight to `cv2.imencode`, which reads it as
  BGR. The bytes are channel-reversed against the JPEG standard, and `cv2.imdecode` reverses them
  back. Everything collected before the marker existed is this.
- **standard** — a conforming RGB JPEG written by `encode_image_bit`, stamped with a JPEG `COM`
  segment holding `XPL-RGB1`. `cv2.imdecode` returns BGR for it, so it needs exactly one swap.

A `COLOR_BGR2RGB` added to "fix" a decode is therefore still always a bug, and now for a sharper
reason than before: the decoded pixels are indistinguishable to the eye — only the marker in the
encoded buffer tells the formats apart — so a caller-side swap is right on at most one of them and
silently wrong on the other. Channel
conversion is allowed **only inside `utils/process_data.py`**, which owns the distinction.

No channel conversion belongs in conversion, training, or eval code. Two exceptions only:

- Medium adapters: `COLOR_RGB2BGR` immediately before `cv2.VideoWriter.write(...)`, and
  `COLOR_BGR2RGB` immediately after `cv2.VideoCapture.read()`.
- A deliberate RGB→BGR conversion for a checkpoint trained on BGR data, opt-in through a documented
  `deploy.yml` key that defaults to RGB (see `input_color_order` in `policy/Dexora_1B`).

## Decoding goes through the shared helpers

`model.py` never decodes. The server decodes every observation it forwards, including for custom
RPCs, so `obs["vision"][<camera>]["color"]` is already an array; adapters only reshape, cast, resize.

Offline code — conversion scripts and training dataloaders — decodes **only** with
`decode_image_bit` from `XPolicyLab.utils.process_data`, and encodes **only** with its inverse
`encode_image_bit`. That is the single supported pair: image bits come in the two byte formats above
and carry further inconsistency in their container layouts from earlier data versions, and only
these functions handle every case. Never hand-roll `cv2.imdecode` / `np.frombuffer` / PIL decoding —
a hand-rolled decoder is right on one byte format and reverses the channels on the other, whichever
way it is written, and the rest trip over the older layouts.

Mechanically, `cv2.imdecode` must not appear outside `utils/process_data.py`, and image bits that
get **stored or published** — trajectory files, converted datasets — must come from
`encode_image_bit`, never from a bare `cv2.imencode`, which omits the marker and so writes a buffer
that can only be read as legacy — channel-reversed on decode if the frame was converted to BGR
before encoding, and even when fed RGB it mints more legacy data that every conforming viewer shows
reversed. The why is in README,
[Standard Data Formats](README.md#decode-only-through-decode_image_bit).

## Paths and dimensions come from the shared helpers

`env_cfg/` lives in the **parent workspace, outside this repo** — `XPolicyLab/env_cfg` does not
exist, so an adapter must never build that path itself.

- **Importable root** in `policy/<POLICY>/model.py` is the parent of the checkout:
  `Path(__file__).resolve().parent.parents[2]`, which is the same directory as
  `Path(__file__).resolve().parents[3]`. Counted from the policy directory, `parents[1]` is the
  checkout itself and `parents[3]` is unrelated; both are bugs.
- **Checkpoints** resolve through `XPolicyLab.utils.checkpoint_resolver` — `resolve_checkpoint_root`,
  or `build_run_dir_name` / `candidate_checkpoint_roots` when an adapter adds its own naming layer —
  never by re-deriving `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
- **Action dimensions** come from `get_robot_action_dim_info(env_cfg_type)` in
  `XPolicyLab.utils.process_data`, never hard-coded and never through a private re-implementation.

A new robot must be registered in **both** robot-info files, or training and evaluation disagree
about action dimensions:

| File | Read by | Keyed by |
| --- | --- | --- |
| `<parent>/env_cfg/robot/_robot_info.json` | `get_robot_action_dim_info()` and `get_action_dim()` in `XPolicyLab.utils.process_data` — runtime and offline conversion | robot name, from `config.robot` in `<parent>/env_cfg/<env_cfg_type>.yml` |
| `utils/robot/_robot_info.json` | `utils/get_action_dim.sh`, called by `train.sh` | `env_cfg_type` |

The two entry points named `get_action_dim` do **not** read the same file. Runtime and conversion
code imports the Python one; the training path — `train.sh`, or a Python training entry it invokes —
uses `utils/get_action_dim.sh`, which delivers the value as a shell variable, runs on stdlib `json`
alone, and works in a standalone checkout that has no outer `env_cfg/` tree.

## deploy.yml

`policy_name` must equal the directory name. Keep the full key set from
`policy/demo_policy/deploy.yml` — including `protocol: ws`, `host` and `port` — even where the
scripts have a default; per-run fields are overridden at launch.
