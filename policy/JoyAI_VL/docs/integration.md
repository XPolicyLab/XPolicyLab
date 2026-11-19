# XPolicyLab integration

The policy is registered as `JoyAI_VL`. It connects the JoyAI-VL Agent to
XPolicyLab's observation/action interface and RoboDojo's episode loop.

## Policy interface

`model.py` inherits the retained bridge and XPolicyLab's `ModelTemplate`. The shared
policy server decodes camera images to RGB arrays before calling `update_obs`.
The setup script adds the policy directory and RoboDojo workspace to the import path.

`deploy.py` supports asynchronous model inference: a `{"hexashell": "wait"}` response
causes another action poll while simulation remains still. Joint-action chunks execute
through the official client, and the final verdict returns through `trial_end`.

## Robot and runtime

The controller uses ARX X5 joint control, one environment per policy server. Robot base
poses come from `ROBODOJO_REPO/env_cfg/robot/dual_x5.yml`; the planner uses the X5 assets,
and the episode step limit comes from the evaluator's task definition.

Docker runs the agent and API proxy. The policy server uses the RoboDojo Isaac/cuRobo
Python environment. A separate SAM3 service supplies object masks and can serve multiple
policy processes.

## Image transport

The XPolicyLab boundary receives decoded RGB arrays. Inside the harness, OpenCV/Pillow
serialize images and masks for the agent and perception service using lossless PNG;
the model API receives JPEG images. RGB/BGR conversions occur in these internal codec
paths. These paths use local codecs rather than XPolicyLab's shared image encoder.

## Verification

The CPU suite checks the actual adapter import, shared RGB decoding, seed/layout
forwarding, configuration quoting, Docker credential routing, API request fields,
reasoning history and asynchronous wait/final-verdict handling with synthetic inputs.
Source and variant hashes, Python parsing and shell syntax are checked separately.

[Reproducibility](reproducibility.md) describes the source and runtime manifests.
