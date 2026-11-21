# Third-party notices

This JoyAI-VL evaluation release is maintained by Joy Future Academy. The root Apache-2.0 license applies to this
distribution subject to the component licenses below; it does not license external model weights.

Paths below are relative to the repository. In a separately installed policy, omit the
`` prefix. The original upstream notice is retained without edits.

| Component | Location / source | License and attribution |
| --- | --- | --- |
| HexaShell Preview and robot implementation | ``; [HexaFuture/HexaShell](https://github.com/HexaFuture/HexaShell/tree/v1.0) | Apache-2.0; root `LICENSE`, plus `vendor/robot15/LICENSE` and `vendor/robot15/NOTICE` inside the policy |
| RoboDojo-derived ARX X5 description / mesh data | `vendor/robot15/hexashell/assets/` | MIT; `licenses/RoboDojo-LICENSE`; Copyright (c) 2026 RoboDojo Contributors |
| Shared perception HTTP wrapper | `services/sam3_server.py` | Derived from HexaShell; Apache-2.0; reference checksum in `metadata/source_manifest.json` |
| Other upstream contributions and design credits | Preserved `vendor/robot15/NOTICE` | Includes PyRUA-Lean and additional project acknowledgments; retained as supplied upstream |

The upstream notice describes a broader HexaShell project. Some paths in that notice are
upstream paths outside this evaluation-only distribution.

External dependencies are installed separately: RoboDojo, NVIDIA Isaac Sim, cuRobo,
Docker, Python packages, and [Meta SAM3](https://github.com/facebookresearch/sam3).
SAM3 model code and checkpoints are subject to the [SAM license](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
and its access terms. Neither the SAM3 model repository nor its weights are bundled here.
JoyAI-VL model weights and API access are also outside this harness release.

Source and packaging history are recorded in `metadata/source_manifest.json` and
`metadata/integration_manifest.json`. This release's changes are recorded
in `metadata/variant_manifest.json`.
