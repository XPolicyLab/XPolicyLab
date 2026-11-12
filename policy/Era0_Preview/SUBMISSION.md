# Era0-Preview submission information

- Contributor: **RobotEra Team**; organization: **RobotEra**.
- Contact: **Liming Zheng**, **liming.zheng@robotera.com**.
- Model type: **VLA**; evaluation type: **simulation**; release scope: **evaluation only**.
- Technical report: planned for public release by **2026-11-18**.
- Model display name: **Era0-Preview**; policy directory: **Era0_Preview**.
- Supported environment/action: **RoboDojo / arx_x5 / ee**.
- Code: inference runtime and evaluation adapter included in this policy directory.
- Assets: [model repository](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo). Matching inference assets are published; download the pinned revision with `bash download_checkpoint.sh`.

## Evaluation evidence

The unmodified official SDK summary is included as [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).

| Item | Current result |
| --- | --- |
| Overview Score | **38.36 ± 0.55** |
| Overview SR | **33.11 ± 0.27%** |
| Complete task–seed cells | **123/126** |
| Saved episodes | **6,265/6,300** |
| SDK media inventory | **159 complete result directories, zero mismatches** |

The intended evaluation uses all SDK default tasks, seeds 0/1/2 and native episode counts. The current result is incomplete: `make_toast_random` lacks 5, 20 and 10 episodes for seeds 0, 1 and 2. Incomplete cells are excluded. These are local test results, not official leaderboard scores.

Installation and launch instructions are in [README.md](README.md). Current validation is recorded in [PREPARATION_CHECKS.md](PREPARATION_CHECKS.md); earlier-release transport and simulator checks remain explicitly labelled historical in their respective reports.

Matching model files and download instructions are supplied. A complete native result remains pending. No additional artifact-release commitment is implied.
