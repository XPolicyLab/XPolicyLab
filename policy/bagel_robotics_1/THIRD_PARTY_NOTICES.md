# Source and artifact notices

Bagel-Robotics-1's method contribution is the pruning/recovery training procedure
and effective selection of task-critical data for an inference-friendly compact
WAM. VPP2 is the base model used to validate that approach. The source notices
below identify reused implementation components and their licenses.

- Base-model inference components adapted from [VPP2](https://github.com/roboterax/video-prediction-policy-2)/FastWAM retain the MIT notice in
  `LICENSE`, copyright 2026 The FastWAM Authors. See `source-provenance.json`
  for the source identities and packaging changes.
- Wan2.1/DiffSynth-derived model components retain Apache-2.0 terms; see
  `licenses/Apache-2.0.txt` and upstream Wan2.1/DiffSynth source notices.
- The XPolicyLab interface follows its upstream license, included in
  `licenses/XPolicyLab.txt`. RoboDojo and simulator assets are not bundled.
- Encoder weights, tokenizer files and model weights have their own upstream
  terms. The code license does not assign a new license to those artifacts.
- The existing preview PR by Soroush Mehraban was consulted as a reference; its
  alternate checkpoint, code and release promises are not this submission.
