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

The official policy server calls `decode_obs_images` before `update_obs`, so the
adapter receives RGB arrays for both legacy and marked XPolicyLab JPEG buffers.
`model.py` and the bridge do not decode official image bits or swap channels.

The agent HTTP interface uses its own lossless PNG pictures. The shared
`roboshell/image_transport.py` encoder and decoder use Pillow and retain RGB
through camera pictures, tool overlays and the array export. The decoder accepts
PNG only; official trajectory JPEGs belong to the XPolicyLab server decoder.
Generic image tools use RGB, with explicit RGB-to-HSV/Lab or grayscale conversion
where their algorithms require those representations. SAM3 receives RGB arrays
and returns grayscale PNG masks; mask values and component pixels are retained.
The model request loop converts the agent's PNG pictures to JPEG with Pillow.

No production policy file calls `cv2.imdecode` or `cv2.imencode`, and none swaps
RGB and BGR. The image test creates one legacy JPEG fixture and consumes it in the
same process to check the official decoder.

## Verification

The CPU suite checks the actual adapter import, shared RGB decoding, seed/layout
forwarding, configuration quoting, Docker credential routing, API request fields,
reasoning history and asynchronous wait/final-verdict handling with synthetic inputs.
Source and variant hashes, Python parsing and shell syntax are checked separately.
Full process checks and commands are recorded in [validation](validation.md).

[Reproducibility](reproducibility.md) describes the source and runtime manifests.
