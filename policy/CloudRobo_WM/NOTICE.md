# Source provenance

The model source is from [CloudRobo-WM-0](https://github.com/Liumouliu/CloudRobo-WM-0): inference commit `08a2f621df09cc4a6e4910fec3602bc422e4afa0` and post-training commit `4c3b80abbd8a3e0029bc509a9fedba45862b9c64`. `vendored_sources.json` records the included files and intentional training changes.

Upstream acknowledges Wan2.2, DINOv3, Depth Anything 3, Fast-WAM and flex-pi. Existing source and URDF notices are retained. These branch snapshots do not supply a LICENSE file; their redistribution terms and the separate model/dataset asset terms need to be supplied by the authors. This notice does not assign a new license to upstream code or assets.

# Local packaging changes

The PR candidate normalizes UTF-8 text line endings to LF and removes only trailing blank lines from `source_inference/src/crwm/processing/eef.py`, `eef_geometry.py` and `kinematics.py`. Their Python ASTs are unchanged. `vendored_sources.json` records the patched content hashes while retaining the original `upstream_sha256` values and source commits. No model, solver, preprocessing or weight values are changed.

The two FK assets in `source_inference/configs/kinematics/` are byte-exact, runtime-SHA-pinned resources. Their upstream trailing whitespace and EOF layout are retained. `.gitattributes` limits the `blank-at-eol` / `blank-at-eof` exceptions to those two paths; source code retains the normal whitespace rules.

The README and result tables report the completed 2026-10-18 native/inprocess validation (88.90% clean, 75.00% randomized, 81.95% average). The separate full WebSocket interface logs in the PR were captured on the immediately preceding tested LF-repaired package. The final candidate differs only in line endings, EOF whitespace, hash bookkeeping and documentation/result tables; the PR includes a source-equivalence and static-check record.
