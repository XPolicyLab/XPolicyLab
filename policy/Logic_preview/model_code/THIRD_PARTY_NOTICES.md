# Third-party notices

- The extracted FastWAM project code retains the MIT notice in [LICENSE](LICENSE),
  copyright 2026 The FastWAM Authors. This extraction changes packaging, limits the
  exposed training/inference modes, makes data/checkpoints portable, and removes
  unrelated experimental entrypoints.
- Wan model components derive from Wan2.1 and the DiffSynth implementation lineage.
  Their upstream Apache-2.0 terms remain applicable. A copy is included in
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Upstream licenses:
  [Wan2.1](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt),
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/LICENSE).
- RoboDojo simulator patch context retains the upstream MIT notice in
  [licenses/RoboDojo-MIT.txt](licenses/RoboDojo-MIT.txt). The simulator and its assets
  are not bundled. Source: [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo).
- The deployment adapter uses the XPolicyLab interface. Its upstream license is
  retained in [licenses/XPolicyLab.txt](licenses/XPolicyLab.txt). Source:
  [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).
- Model weights and datasets are distributed separately under their own terms;
  the code's license does not assign licenses to those artifacts.

- The vendored local LeRobot v2.1 dataset reader retains the Hugging Face team's
  Apache-2.0 notices in its source files. See [Apache-2.0](licenses/Apache-2.0.txt).

Source publication: [arXiv:2610.10270](https://arxiv.org/abs/2610.10270). This distribution contains the inference modules and import-name compatibility for the released checkpoint.
